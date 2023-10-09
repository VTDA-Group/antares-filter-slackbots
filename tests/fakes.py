import collections
import json
import operator
import time
import uuid
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from astropy.coordinates import Angle, SkyCoord

from antares.adapters.concurrency import AbstractDistributedLock
from antares.adapters.messages import AbstractMessagePublicationService
from antares.adapters.notifications import AbstractNotificationService
from antares.adapters.repository.base import (
    AbstractAlertRepository,
    AbstractAnnouncementRepository,
    AbstractBlobRepository,
    AbstractCatalogObjectRepository,
    AbstractFilterRepository,
    AbstractFilterRevisionRepository,
    AbstractGravWaveRepository,
    AbstractJwtBlocklistRepository,
    AbstractLocusAnnotationRepository,
    AbstractLocusRepository,
    AbstractUserRepository,
    AbstractWatchListRepository,
    AbstractWatchObjectRepository,
    ListQueryFilters,
)
from antares.adapters.repository.bigtable.catalog import (
    BigtableCatalogObjectTableDescription,
    check_properties_have_minimal_columns,
)
from antares.domain.models import (
    Alert,
    Announcement,
    Blob,
    Catalog,
    CatalogObject,
    Filter,
    FilterContext,
    FilterExecutable,
    FilterRevision,
    FilterRevisionStatus,
    GravWaveNotice,
    GravWaveNoticeTypes,
    JwtRecord,
    Locus,
    LocusAnnotation,
    Survey,
    User,
    WatchList,
    WatchObject,
)
from antares.entrypoints.pipeline.stages.ingest_alert_packet import AbstractAlertPacket
from antares.entrypoints.pipeline.stages.load_alert_packet import (
    DecatAlertPacket,
    ZtfAlertPacket,
)
from antares.entrypoints.pipeline.stages.load_alert_packet.decat import (
    DecatEventObject_0_11,
    DecatEventSource_0_11,
)
from antares.entrypoints.pipeline.stages.load_alert_packet.ztf import (
    ZtfEvent_3_3,
    ZtfEventCandidate_3_3,
    ZtfEventPrvCandidate_3_3,
)
from antares.exceptions import KeyViolationException


class FakeAlertPacket(AbstractAlertPacket):
    def __init__(self, alerts: list[Alert], centroid: SkyCoord, is_good: bool = True):
        super().__init__()
        self._alerts = alerts
        self._centroid = centroid
        self._is_good = is_good

    @property
    def alerts(self) -> list[Alert]:
        return self._alerts

    @property
    def triggering_alert(self) -> Alert:
        return self.alerts[-1]

    @property
    def location(self) -> SkyCoord:
        return self._centroid

    @property
    def is_good(self) -> bool:
        return self._is_good

    def get_associated_locus(
        self, locus_repository: AbstractLocusRepository
    ) -> Optional[Locus]:
        candidates = locus_repository.list_by_cone_search(self.location, Angle("1s"))
        candidates = list(candidates)
        candidates.sort(key=lambda locus: locus.location.separation(self.location))
        try:
            return next(iter(candidates))
        except StopIteration:
            return None


class FakeDistributedLock(AbstractDistributedLock):
    def __init__(
        self,
        database: set[tuple[str, Optional[float]]],
        id_: str,
        ttl: Optional[float] = None,
    ):
        if not isinstance(id_, str):
            raise ValueError("Lock `id_` must be of type `str`")
        self._database = database
        self._ttl = ttl
        self._id = id_

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        start_time = time.perf_counter()
        key_found = any(self._id == id_ for (id_, ttl) in self._database)
        while key_found:
            ttl = next(ttl for (id_, ttl) in self._database if self._id == id_)
            if ttl and ttl < time.perf_counter():
                self.release()
            if not blocking and (ttl is None or ttl > time.perf_counter()):
                return False
            if blocking and (time.perf_counter() - start_time) > timeout:
                return False
            key_found = any(self._id == id_ for (id_, ttl) in self._database)
        ttl: Optional[float] = None
        if self._ttl:
            ttl = time.perf_counter() + self._ttl
        self._database.add((self._id, ttl))
        self._locked = True
        return True

    def release(self) -> None:
        remove_set = set()
        for id_, ttl in self._database:
            if self._id == id_:
                remove_set.add((id_, ttl))
        if len(remove_set):
            self._database -= remove_set
            self._locked = False

    def locked(self):
        return self._locked


class FakeGravWaveRepository(AbstractGravWaveRepository):
    def __init__(self, notices: list[GravWaveNotice] = None):
        self._notices = notices or []
        self.current_notices = []

    def add(self, notice: GravWaveNotice) -> None:
        duplicates = [
            n
            for n in self._notices
            if n.gracedb_id == notice.gracedb_id
            and n.notice_datetime == notice.notice_datetime
        ]
        if duplicates:
            raise KeyViolationException("Duplicate entry")
        if notice.id is None:
            notice.id = len(self._notices) + 1
        self._notices.append(notice)

    def get(self, id_: str) -> Optional[GravWaveNotice]:
        try:
            return next(notice for notice in self._notices if notice.id == id_)
        except StopIteration:
            return None

    def get_id(
        self, gracedb_id: str, notice_datetime: datetime
    ) -> Optional[GravWaveNotice]:
        try:
            return next(
                notice.id
                for notice in self._notices
                if notice.gracedb_id == gracedb_id
                and notice.notice_datetime == notice_datetime
            )
        except StopIteration:
            return None

    def get_current_notices(self):
        return self.current_notices

    def get_latest_active_notice_ids(self):
        return set()


class FakeLocusRepository(AbstractLocusRepository):
    def __init__(self, loci: list[Locus] = None):
        self._loci = loci or []

    def get(self, id_: str) -> Optional[Locus]:
        try:
            return next(locus for locus in self._loci if locus.id == id_)
        except StopIteration:
            return None

    def list_by_cone_search(self, location: SkyCoord, radius: Angle) -> list[Locus]:
        return [
            locus
            for locus in self._loci
            if locus.location.separation(location) <= radius
        ]

    def add(self, locus: Locus) -> None:
        locus_ids = [l.id for l in self._loci]
        if locus.id in locus_ids:
            raise KeyViolationException
        self._loci.append(locus)

    def update(self, locus: Locus) -> None:
        pass


class FakeAnnouncementRepository(AbstractAnnouncementRepository):
    def __init__(self, announcements: list[Announcement] = None):
        self._announcements = announcements or []
        self._pk_auto_increment = 0

    def get(self, id_: str) -> Optional[Announcement]:
        try:
            return next(a for a in self._announcements if a.id == id_)
        except StopIteration:
            return None

    def list(
        self, query_filters: Optional[ListQueryFilters] = None
    ) -> list[Announcement]:
        query_filters = query_filters or []
        results = (a for a in self._announcements)
        for query_filter in query_filters:
            op = getattr(operator, query_filter["op"])
            results = (
                a
                for a in results
                if op(getattr(a, query_filter["field"]), query_filter["value"])
            )
        return results

    def add(self, announcement: Announcement) -> None:
        announcement.id = announcement.id or self._pk_auto_increment
        self._pk_auto_increment += 1
        announcement_ids = [a.id for a in self._announcements]
        if announcement.id in announcement_ids:
            raise KeyViolationException
        self._announcements.append(announcement)

    def update(self, announcement: Announcement) -> None:
        pass


class FakeLocusAnnotationRepository(AbstractLocusAnnotationRepository):
    def __init__(self, locus_annotations: list[LocusAnnotation] = None):
        self._locus_annotations = locus_annotations or []

    def get(self, id_: str) -> Optional[LocusAnnotation]:
        try:
            return next(l for l in self._locus_annotations if l.id == id_)
        except StopIteration:
            return None

    def add(self, locus_annotation: LocusAnnotation) -> None:
        locus_annotation_ids = [l.id for l in self._locus_annotations]
        if locus_annotation.id in locus_annotation_ids:
            raise KeyViolationException
        self._locus_annotations.append(locus_annotation)

    def update(self, locus_annotation: LocusAnnotation) -> None:
        pass

    def list_by_owner_id(self, owner_id) -> Iterable[LocusAnnotation]:
        return [l for l in self._locus_annotations if l.owner_id == owner_id]


class FakeAlertRepository(AbstractAlertRepository):
    def __init__(self, alerts: list[tuple[str, Alert]] = None):
        self._alerts = alerts or []

    def get(self, id_: str) -> Optional[Alert]:
        try:
            return next(alert for (_, alert) in self._alerts if alert.id == id_)
        except StopIteration:
            return None

    def add(self, alert: Alert, locus_id: str) -> None:
        self._alerts.append((locus_id, alert))

    def list_by_locus_id(self, locus_id: str) -> Iterable[Alert]:
        return (
            alert
            for (associated_locus_id, alert) in self._alerts
            if associated_locus_id == locus_id
        )


class FakeCatalogObjectRepository(AbstractCatalogObjectRepository):
    def __init__(
        self,
        catalog_objects: Optional[list[CatalogObject]] = None,
        catalogs: Optional[list[Catalog]] = None,
    ) -> None:
        super(FakeCatalogObjectRepository, self).__init__(catalogs or [])
        self._catalog_objects = catalog_objects if catalog_objects else []

    def list_by_catalog_id(self, catalog_id: str) -> Iterable[CatalogObject]:
        yield from (
            catalog_object
            for catalog_object in self._catalog_objects
            if catalog_object.catalog_id == catalog_id
        )

    def list_by_location(self, location: SkyCoord) -> Iterable[CatalogObject]:
        return [
            catalog_object
            for catalog_object in self._catalog_objects
            if catalog_object.location.separation(location) <= catalog_object.radius
        ]

    def add(self, catalog_object: CatalogObject) -> None:
        self._catalog_objects.append(catalog_object)


class FakeFilterRepository(AbstractFilterRepository):
    """
    Test

    Parameters
    ----------
    filters: Optional[list[Filter]]
    """

    def __init__(self, filters: Optional[list[Filter]] = None):
        self._filters = filters if filters else []

    def add(self, filter_: Filter):
        self._filters.append(filter_)

    def get(self, id_: int) -> Optional[Filter]:
        try:
            return next(filter_ for filter_ in self._filters if filter_.id == id_)
        except StopIteration:
            return None

    def list_by_owner_id(self, owner_id) -> Iterable[Filter]:
        return [f for f in self._filters if f.owner_id == owner_id]

    def list(self) -> Iterable[Filter]:
        return [f for f in self._filters]

    def get_filter_executables(
        self, filter_revision_repository: AbstractFilterRevisionRepository
    ):
        return [
            filter_revision_repository.get_filter_executable(filter_)
            for filter_ in self.list()
            if filter_.enabled
        ]

    def update(self, filter_: Filter) -> None:
        pass


class FakeFilterRevisionRepository(AbstractFilterRevisionRepository):
    def __init__(self, filter_revisions: Optional[list[FilterRevision]] = None):
        self._filter_revisions = filter_revisions if filter_revisions else []

    def get(self, id_: int) -> Optional[FilterRevision]:
        try:
            return next(
                filter_revision
                for filter_revision in self._filter_revisions
                if filter_revision.id == id_
            )
        except StopIteration:
            return None

    def get_filter_executable(self, filter_: Filter) -> tuple[Filter, FilterExecutable]:
        filter_revision = self.get(filter_.enabled_filter_revision_id)
        return filter_, filter_revision.to_filter_executable()

    def list_by_filter_id(self, filter_id: int) -> list[FilterRevision]:
        return [
            filter_revision
            for filter_revision in self._filter_revisions
            if filter_revision.filter_id == filter_id
        ]

    def add(self, filter_revision: FilterRevision) -> None:
        self._filter_revisions.append(filter_revision)

    def update(self, filter_revision: FilterRevision) -> None:
        pass


class FakeJwtBlocklistRepository(AbstractJwtBlocklistRepository):
    def __init__(self, jwts: Optional[list[JwtRecord]] = None):
        self._jwts = jwts if jwts else []

    def add(self, jwt: JwtRecord) -> None:
        self._jwts.append(jwt)

    def get_by_jti(self, jti: str) -> Optional[JwtRecord]:
        try:
            return next(jwt for jwt in self._jwts if jwt.jti == jti)
        except StopIteration:
            return None


class FakeUserRepository(AbstractUserRepository):
    def __init__(self, users: Optional[list[User]] = None):
        self._users = users if users else []

    def list(
        self, filtered_ids: Optional[Iterable[uuid.UUID]] = None
    ) -> Iterable[User]:
        if filtered_ids:
            yield from [user for user in self._users if user.id in filtered_ids]
        else:
            yield from self._users

    def update(self, user: User) -> None:
        pass

    def get(self, id_: uuid.UUID) -> Optional[User]:
        try:
            return next(user for user in self._users if user.id == id_)
        except StopIteration:
            return None

    def get_by_username(self, username: str) -> Optional[User]:
        try:
            return next(user for user in self._users if user.username == username)
        except StopIteration:
            return None

    def add(self, user: User) -> None:
        if any(user.username == u.username for u in self._users):
            raise KeyViolationException
        self._users.append(user)


class FakeWatchListRepository(AbstractWatchListRepository):
    def __init__(self, watch_lists: Optional[Iterable[WatchList]] = None) -> None:
        self._watch_lists = set(watch_lists) if watch_lists else set()

    def add(self, watch_list: WatchList) -> None:
        self._watch_lists.add(watch_list)

    def get(self, id_: uuid.UUID) -> Optional[WatchList]:
        try:
            return next(
                watch_list
                for watch_list in self._watch_lists
                if str(watch_list.id) == str(id_)
            )
        except StopIteration:
            return None

    def list_by_owner_id(self, owner_id: UUID) -> Iterable[WatchList]:
        yield from (
            watch_list
            for watch_list in self._watch_lists
            if watch_list.owner_id == owner_id
        )

    def delete(self, id_: UUID):
        watch_list = self.get(id_)
        self._watch_lists.remove(watch_list)


class FakeWatchObjectRepository(AbstractWatchObjectRepository):
    def __init__(self, watch_objects: Optional[Iterable[WatchObject]] = None) -> None:
        self._watch_objects = set(watch_objects) if watch_objects else set()

    def add(self, watch_object: WatchObject) -> None:
        self._watch_objects.add(watch_object)

    def get(self, id_: uuid.UUID) -> Optional[WatchObject]:
        try:
            return next(wo for wo in self._watch_objects if wo.id == id_)
        except StopIteration:
            return None

    def list_by_watch_list_id(self, watch_list_id: uuid.UUID) -> Iterable[WatchObject]:
        yield from (
            wo for wo in self._watch_objects if wo.watch_list_id == watch_list_id
        )

    def list_by_location(self, location: SkyCoord) -> Iterable[WatchObject]:
        return [
            wo
            for wo in self._watch_objects
            if wo.location.separation(location) < wo.radius
        ]


class FakeBlobRepository(AbstractBlobRepository):
    def __init__(self, blobs: Optional[list[Blob]] = None):
        self._blobs = blobs if blobs else []

    def list(self) -> Iterable[Blob]:
        yield from self._blobs

    def update(self, blob: Blob) -> None:
        pass

    def get(self, id_: int) -> Optional[Blob]:
        try:
            return next(blob for blob in self._blobs if blob.id == id_)
        except StopIteration:
            return None

    def add(self, blob: Blob) -> None:
        self._blobs.append(blob)


class FakeMessagePublicationService(AbstractMessagePublicationService):
    def __init__(self):
        self.published: dict[str, list[Any]] = collections.defaultdict(list)

    def publish(self, destination: str, message: Any):
        self.published[destination].append(message)


class FakeNotificationService(AbstractNotificationService):
    def __init__(self):
        self.sent: list[tuple[str, Any]] = []

    def send(self, destination: str, message: Any):
        self.sent.append((destination, message))


def build_ztf_alert(**kwargs) -> Alert:
    defaults = {
        "alert_id": "alert-001",
        "location": SkyCoord("0d 0d"),
        "survey": Survey.SURVEY_ZTF,
        "properties": {},
        "normalized_properties": {
            "ant_mjd": kwargs.get("mjd", 59000),
            "ant_mag": 16.0,
            "ant_magerr": 0.0,
            "ant_maglim": 0.0,
            "ant_survey": 1,
            "ant_ra": 0.0,
            "ant_dec": 0.0,
            "ant_passband": "g",
        },
        "mjd": 59000.0,
        "created_at": datetime.utcnow(),
    }
    defaults.update(**kwargs)
    return Alert(**defaults)


def build_alert(**kwargs) -> Alert:
    defaults = {
        "id": "alert-001",
        "location": SkyCoord("0d 0d"),
        "survey": Survey.SURVEY_ZTF,
        "properties": {},
        "normalized_properties": {
            "ant_mjd": kwargs.get("mjd", 59000),
            "ant_mag": 16.0,
            "ant_magerr": 0.0,
            "ant_maglim": 0.0,
            "ant_survey": 1,
            "ant_ra": 0.0,
            "ant_dec": 0.0,
            "ant_passband": "g",
        },
        "mjd": kwargs.get("mjd", 59000.0),
        "created_at": datetime.utcnow(),
    }
    defaults.update(**kwargs)
    return Alert(**defaults)


def build_alerts(n: int, **kwargs) -> list[Alert]:
    mjd = kwargs.pop("mjd", 59000.0)
    return [
        build_alert(
            id=f"alert-{i:03d}",
            mjd=(mjd + i),
            **kwargs,
        )
        for i in range(n)
    ]


def build_notice(**kwargs) -> GravWaveNotice:
    with open(
        "test/data/e2e/grav_wave/MS181101ab-earlywarning.json", "rt", encoding="utf8"
    ) as notice_file:
        real_notice = json.load(notice_file)

    real_notice.update(**kwargs)
    notice = GravWaveNotice.from_gcn(real_notice)
    notice.event_datetime = kwargs.get(
        "event_datetime", datetime.utcnow() - timedelta(days=1)
    )
    notice.notice_datetime = kwargs.get("notice_datetime", datetime.utcnow())
    return notice


def build_retraction_notice(**kwargs) -> GravWaveNotice:
    with open(
        "test/data/e2e/grav_wave/MS181101ab-retraction.json", "rt", encoding="utf8"
    ) as notice_file:
        real_notice = json.load(notice_file)
    real_notice.update(**kwargs)
    return GravWaveNotice.from_gcn(real_notice)


def build_locus(**kwargs) -> Locus:
    defaults = {
        "id": "locus-001",
        "location": SkyCoord("0d 0d"),
        "properties": {},
        "catalogs": set(),
        "tags": set(),
        "watch_object_matches": set(),
    }
    defaults.update(**kwargs)
    return Locus(**defaults)


def build_catalog_object(**kwargs) -> CatalogObject:
    defaults = {
        "id": "1",
        "catalog_id": "1",
        "name": "Catalog Object",
        "catalog_name": "catalog-001",
        "location": SkyCoord("0d 0d"),
        "radius": Angle("0d"),
        "properties": {
            "objid": 44183,
            "creationdate": datetime(2019, 8, 29, 17, 41, 26),
            "declination": 2.9090313272727,
            "discmagfilter": 111.0,
            "discoverydate": datetime(2019, 8, 24, 9, 1, 26),
            "discoverymag": 19.0,
            "internal_names": "ZTF19abrmlxu",
            "lastmodified": datetime(2019, 8, 29, 17, 41, 26),
            "name": "2019oym",
            "name_prefix": "SN",
            "ra": 14.973595209091,
            "redshift": 0.065,
            "reporting_group": "ZTF",
            "reporting_groupid": "48",
            "reports": None,
            "source_group": "ZTF",
            "source_groupid": "48",
            "time_received": datetime(2019, 8, 29, 17, 41, 25),
            "type": "SN Ia",
            "typeid": "3",
        },
    }
    defaults.update(**kwargs)
    return CatalogObject(**defaults)


def build_filter(**kwargs) -> Filter:
    defaults = {
        "id": 1,
        "name": "Test Filter",
        "description": "Test filter description",
        "enabled_filter_revision_id": None,
        "public": True,
        "owner_id": uuid.UUID(int=1),
    }
    defaults.update(**kwargs)
    return Filter(**defaults)


def build_filter_context(
    locus: Optional[Locus] = None, alerts: Optional[list[Alert]] = None
) -> FilterContext:
    from antares.domain.models.filter import (
        build_filter_context as build_filter_context_from_models,
    )

    if alerts is None:
        alerts = [build_alert(mjd=59000 + i) for i in range(5)]
    if locus is None:
        locus = build_locus(id="locus-001", properties={"foo": "bar"})
    catalog_objects = [build_catalog_object()]
    return build_filter_context_from_models(locus, alerts, catalog_objects)


def build_filter_revision(filter_: Filter, **kwargs) -> FilterRevision:
    # "callable_": lambda *args, **kwargs: None,
    # "output_specification": FilterOutputSpecification(properties=[], tags=[]),
    defaults = {
        "id": 1,
        "filter_id": filter_.id,
        "status": FilterRevisionStatus.REVIEWED,
        "code": "from antares.devkit.filter import Filter\n\nclass TestFilter(Filter):\n\tdef run(self, locus):\n\t\tpass",
        "comment": "A test filter revision",
    }
    defaults.update(**kwargs)
    return FilterRevision(**defaults)


def build_filter_revision_crashes_on_setup(filter_: Filter, **kwargs) -> FilterRevision:
    code = """
import antares.devkit as dk
class FilterWithSetup(dk.Filter):
    ERROR_SLACK_CHANNEL = None
    def setup(self):
        raise Exception("Bad Setup")
    def run(self, locus):
        pass
    """
    return build_filter_revision(filter_, code=code, **kwargs)


def build_filter_revision_raises_exception(filter_: Filter, **kwargs) -> FilterRevision:
    code = """
from antares.devkit.filter import Filter
class TestFilter(Filter):
    def run(self, locus):
        raise Exception("Oh No!")
    """
    return build_filter_revision(filter_, code=code, **kwargs)


def build_filter_revision_sets_tag(
    filter_: Filter, tag: str, **kwargs
) -> FilterRevision:
    code = f"""
from antares.devkit.filter import Filter
class TestFilter(Filter):
    OUTPUT_TAGS = [{{"name": "{tag}", "description": "Test tag"}}]
    def run(self, locus):
        locus.tag("{tag}")
    """
    return build_filter_revision(filter_, code=code, **kwargs)


def build_filter_revision_sets_property(
    filter_: Filter, key: str, value: str, **kwargs
) -> FilterRevision:
    code = f"""
from antares.devkit.filter import Filter
class TestFilter(Filter):
    OUTPUT_LOCUS_PROPERTIES = [{{"name": "{key}", "type": 'str', "description": "Test property"}}]
    def run(self, locus):
        locus.properties["{key}"] = "{value}"
    """
    return build_filter_revision(filter_, code=code, **kwargs)


def build_watch_list(**kwargs) -> WatchList:
    defaults = {
        "owner_id": uuid.UUID(int=1),
        "name": "Watch List",
        "description": "Fake watch list",
    }
    defaults.update(**kwargs)
    return WatchList(**defaults)


def build_watch_object(**kwargs) -> WatchObject:
    defaults = {
        "location": SkyCoord("0d 0d"),
        "radius": Angle("0d"),
        "name": "test watch object",
        "watch_list_id": uuid.uuid4(),
    }
    defaults.update(**kwargs)
    return WatchObject(**defaults)


def build_ztf_event_prv_candidate(**kwargs) -> ZtfEventPrvCandidate_3_3:
    defaults = {
        "candid": "ztf-candid-001",
        "jd": 2000,
        "fid": 1,
        "pid": 123,
        "programid": 456,
        "isdiffpos": "t",
        "ra": 0.0,
        "dec": 0.0,
        "magpsf": 0.0,
        "sigmapsf": 0.0,
        "rbversion": "1.0",
        "drbversion": "1.0",
        "ranr": 0.0,
        "decnr": 0.0,
        "ndethist": 1,
        "ncovhist": 1,
        "rfid": 123,
        "jdstartref": 1000,
        "jdendref": 2000,
        "nframesref": 1,
        "diffmaglim": 16.0,
    }
    defaults.update(kwargs)
    return ZtfEventPrvCandidate_3_3(**defaults)


def build_ztf_event_candidate(**kwargs) -> ZtfEventCandidate_3_3:
    defaults = {
        "candid": "ztf-candid-001",
        "jd": 2000,
        "fid": 1,
        "pid": 123,
        "programid": 456,
        "isdiffpos": "t",
        "ra": 0.0,
        "dec": 0.0,
        "magpsf": 0.0,
        "sigmapsf": 0.0,
        "rbversion": "1.0",
        "drbversion": "1.0",
        "ranr": 0.0,
        "decnr": 0.0,
        "ndethist": 1,
        "ncovhist": 1,
        "rfid": 123,
        "jdstartref": 1000,
        "jdendref": 2000,
        "nframesref": 1,
        "nmtchps": 1,
        "nmatches": 1,
        "diffmaglim": 16.0,
    }
    defaults.update(kwargs)
    return ZtfEventCandidate_3_3(**defaults)


def build_ztf_event(
    candidate_kwargs: Optional[dict] = None,
    prv_candidate_kwargs: Optional[list[dict]] = None,
    **kwargs,
) -> ZtfEvent_3_3:
    candidate_kwargs = candidate_kwargs or {}
    prv_candidate_kwargs = prv_candidate_kwargs or []
    defaults = {
        "schemavsn": "3.3",
        "publisher": "ztf",
        "objectId": "ztf-object-001",
        "candid": "ztf-candid-001",
        "candidate": build_ztf_event_candidate(**candidate_kwargs),
        "prv_candidates": [
            build_ztf_event_prv_candidate(**kwargs) for kwargs in prv_candidate_kwargs
        ],
        "cutoutScience": None,
        "cutoutTemplate": None,
        "cutoutDifference": None,
    }
    defaults.update(kwargs)
    return ZtfEvent_3_3(**defaults)


def build_ztf_alert_packet(
    candidate_kwargs: Optional[dict] = None,
    prv_candidate_kwargs: Optional[list[dict]] = None,
    locus_association_radius: Optional[Angle] = Angle("1.5s"),
    locus_association_search_radius: Optional[Angle] = Angle("5s"),
    **kwargs,
) -> ZtfAlertPacket:
    return ZtfAlertPacket(
        build_ztf_event(candidate_kwargs, prv_candidate_kwargs, **kwargs),
        locus_association_radius=locus_association_radius,
        locus_association_search_radius=locus_association_search_radius,
    )


def build_decat_source(**kwargs) -> DecatEventSource_0_11:
    defaults = {
        "sourceid": 123,
        "ra": 0.0,
        "dec": 0.0,
        "mag": 15.0,
        "magerr": 1.0,
        "flux": 10.0,
        "fluxerr": 0.5,
        "rb": 1.0,
        "rbcut": 0.5,
        "propid": "proposal-1",
        "filter": "g DECam SDSS c0001 4720.0 1520.0",
        "mjd": 59000.0,
        "is_stack": False,
        "ccdnum": 18,
        "sciurl": "https://portal.nersc.gov/cfs/m2218/decat/something.fits.fz",
        "refurl": "https://portal.nersc.gov/cfs/m2218/decat/something.fits.fz",
        "diffurl": "https://portal.nersc.gov/cfs/m2218/decat/something.fits.fz",
    }
    defaults.update(kwargs)
    return DecatEventSource_0_11(**defaults)


def build_decat_event(
    object_kwargs: Optional[dict] = None,
    sources_kwargs: Optional[list[dict]] = None,
) -> DecatEventObject_0_11:
    defaults = {
        "objectid": 100,
        "ra": 0.0,
        "dec": 0.0,
        "gallong": 0.0,
        "gallat": 0.0,
        "tdsic": "2021-11-19T18:04:55.000000",
        "ls_check": False,
    }
    defaults.update(
        {
            **(object_kwargs or {}),
            "sources": [
                build_decat_source(**source_kwargs)
                for source_kwargs in (sources_kwargs or [])
            ],
        }
    )
    return DecatEventObject_0_11(**defaults)


def build_decat_alert_packet(
    object_kwargs: Optional[dict] = None,
    sources_kwargs: Optional[list[dict]] = None,
    locus_association_radius: Optional[Angle] = Angle("1.5s"),
    locus_association_search_radius: Optional[Angle] = Angle("5s"),
    **kwargs,
) -> DecatAlertPacket:
    return DecatAlertPacket(
        build_decat_event(object_kwargs, sources_kwargs),
        locus_association_radius=locus_association_radius,
        locus_association_search_radius=locus_association_search_radius,
    )


def build_user(**kwargs):
    defaults = {
        "name": "Nic Wolf",
        "username": "nwolf",
        "email": "nic.wolf@noirlab.edu",
    }
    defaults.update(kwargs)
    return User(**defaults)


def build_bigtable_catalog_object_table_description(
    **kwargs,
) -> BigtableCatalogObjectTableDescription:
    defaults = {
        "table": "catalog_001",
        "column_family": "C",
        "catalog_id": 1,
        "display_name": "Catalog 001",
        "enabled": True,
        "ra_column": "ra_deg",
        "dec_column": "dec_deg",
        "object_id_column": "id",
        "object_id_type": "int",
        "object_name_column": "name",
        "radius": None,
        "radius_column": None,
        "radius_unit": None,
    }
    defaults.update(**kwargs)
    return BigtableCatalogObjectTableDescription(**defaults)


def build_catalog_object_using_table_description(
    properties: dict, table_description: BigtableCatalogObjectTableDescription
) -> CatalogObject:
    # Location will always be in degree
    check_properties_have_minimal_columns(properties, table_description)
    properties[table_description["object_id_column"]] = int(
        properties[table_description["object_id_column"]]
    )
    radius = None
    if table_description["radius_column"]:
        radius_property = properties.get(table_description["radius_column"])
        if radius_property is None:
            radius = None
        if isinstance(radius_property, Angle):
            radius = radius_property
            properties[table_description["radius_column"]] = float(radius.value)
        else:
            radius = Angle(
                properties.get(table_description["radius_column"]),
                unit=table_description["radius_unit"],
            )
    catalog_object = CatalogObject(
        id=str(properties[table_description["object_id_column"]]),
        catalog_id=str(table_description["catalog_id"]),
        catalog_name=table_description["table"],
        location=SkyCoord(
            ra=properties[table_description["ra_column"]],
            dec=properties[table_description["dec_column"]],
            unit="degree",
        ),
        radius=radius,
        properties=properties,
        name=properties[table_description["object_name_column"]],
    )
    return catalog_object
