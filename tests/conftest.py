import json
import os
import uuid

import antares.bootstrap as real_bootstrap
import antares.exceptions
import antares.external.elasticsearch
import pytest
import sqlalchemy_utils
from antares.adapters.repository import (
    SqlAlchemyFilterRepository,
    SqlAlchemyUserRepository,
)
from antares.adapters.repository.bigtable.schemas import SCHEMAS as BIGTABLE_SCHEMAS
from antares.adapters.repository.cassandra import SCHEMAS as CASSANDRA_SCHEMAS
from antares.adapters.repository.cassandra import (
    CassandraCatalogObjectRepository,
    CassandraCatalogObjectTableDescription,
)
from antares.adapters.repository.sqlalchemy import metadata
from antares.bootstrap import bootstrap
from antares.domain.models import Filter, User
from antares.external.bigtable.bigtable import (
    replace_emulator_channel_fn,
    wait_for_bigtable,
)
from antares.external.cassandra import cassandra
from antares.external.cassandra.cassandra import wait_for_cassandra
from antares.external.elasticsearch.ingest import create_index, delete_index
from antares.external.sql.bootstrap import drop_main_db
from antares.external.sql.engine import get_engine
from antares.logging import log
from antares.settings import Container
from cassandra.cqlengine import columns
from cassandra.cqlengine import connection as cassandra_connection
from cassandra.cqlengine.management import (
    create_keyspace_simple,
    drop_keyspace,
    sync_table,
)
from cassandra.cqlengine.models import Model
from google.cloud import bigtable
from sqlalchemy.orm import sessionmaker


from .fakes import (
    FakeAlertRepository,
    FakeAnnouncementRepository,
    FakeBlobRepository,
    FakeCatalogObjectRepository,
    FakeDistributedLock,
    FakeFilterRepository,
    FakeFilterRevisionRepository,
    FakeGravWaveRepository,
    FakeJwtBlocklistRepository,
    FakeLocusAnnotationRepository,
    FakeLocusRepository,
    FakeMessagePublicationService,
    FakeNotificationService,
    FakeUserRepository,
    FakeWatchListRepository,
    FakeWatchObjectRepository,
)


# NOTE: Destry would prefer that the Repos were created here instead
# of having to bootstrap, but creating the cassandra session requires
# dependency injection to be wired up currently


# @pytest.fixture(scope="module")  # so that it's only called once
# def container():
#     # Next 2 lines are to quiet the warning from the cassandra driver
#     if os.getenv("CQLENG_ALLOW_SCHEMA_MANAGEMENT") is None:
#         os.environ["CQLENG_ALLOW_SCHEMA_MANAGEMENT"] = "1"
#     if os.getenv("ANTARES_CONFIG_YAML") is None:
#         os.environ[
#             "ANTARES_CONFIG_YAML"
#         ] = "/home/delucchi/git/antares/antares/default.yaml"
#     container = bootstrap()
#     yield container
#     container.unwire()


@pytest.fixture
def int_alert_repo():
    # container.config.from_dict(generate_container_config_dict("cassandra"))
    # bootstrap_cassandra(container)
    # cassandra.init()

    # yield container.alert_repository()
    # drop_keyspace(container.config.repositories.alert.cassandra.parameters.keyspace())
    yield FakeAlertRepository()


@pytest.fixture
def int_locus_repo(container):
    container.config.from_dict(generate_container_config_dict("cassandra"))
    bootstrap_cassandra(container)
    cassandra.init()

    yield container.locus_repository()
    drop_keyspace(container.config.repositories.locus.cassandra.parameters.keyspace())


@pytest.fixture
def int_watch_list_repo(container):
    container.config.from_dict(generate_container_config_dict("cassandra"))
    bootstrap_cassandra(container)
    cassandra.init()

    yield container.watch_list_repository()
    drop_keyspace(
        container.config.repositories.watch_list.cassandra.parameters.keyspace()
    )


@pytest.fixture
def int_watch_object_repo(container):
    container.config.from_dict(generate_container_config_dict("cassandra"))
    bootstrap_cassandra(container)
    cassandra.init()

    yield container.watch_object_repository()
    drop_keyspace(
        container.config.repositories.watch_object.cassandra.parameters.keyspace()
    )


@pytest.fixture
def int_catalog_object_repo(container):
    bootstrap_cassandra(container)
    cassandra.init()
    yield container.catalog_object_repository()
    drop_keyspace(container.config.repositories.catalog.cassandra.parameters.keyspace())


@pytest.fixture
def catalog_object_repository(request, session):
    """
    The setup of this repository is a bit more complicated than the others because of
    the way that catalogs are configured in the system.
    """
    marker = request.node.get_closest_marker("with_cassandra_catalogs")
    catalog_tables = []
    if marker:
        catalog_tables: list[CassandraCatalogObjectTableDescription] = [
            model for model in marker.args
        ]
        from cassandra.cqlengine.models import Model, columns

        for catalog_table in catalog_tables:
            # We dynamically create the catalog table schemas in this fixture
            column_definitions = {
                catalog_table["object_id_column"]: columns.BigInt(partition_key=True),
                catalog_table["object_name_column"]: columns.Text(),
                catalog_table["ra_column"]: columns.Float(),
                catalog_table["dec_column"]: columns.Float(),
            }
            if catalog_table["radius_column"]:
                column_definitions[catalog_table["radius_column"]] = columns.Float()
            catalog_schema = type(
                "CatalogSchema",
                (Model,),
                {
                    "__table_name__": catalog_table["table"],
                    **column_definitions,
                },
            )
            sync_table(catalog_schema)
    yield CassandraCatalogObjectRepository(session, catalog_tables)


# ======================================================================================
# Models for populating the databases
# ======================================================================================

users = [
    User(
        id=uuid.UUID(int=1),
        name="admin",
        username="admin",
        email="admin@noirlab.edu",
        admin=True,
    ),
    User(
        id=uuid.UUID(int=2),
        name="staff",
        username="staff",
        email="staff@noirlab.edu",
        staff=True,
    ),
    User(id=uuid.UUID(int=3), name="user", username="user", email="user@noirlab.edu"),
]
for user in users:
    user.set_password("password")

filters = [
    Filter(
        id=1,
        description="A test filter",
        name="Test Filter",
        public=True,
        owner_id=uuid.UUID(int=1),
    )
]


# ======================================================================================
# Bootstrap Code
# ======================================================================================
class CassandraCatalogSchema(Model):
    __table_name__ = "catalog_table"
    object_id = columns.Integer(partition_key=True)
    ra = columns.Float()
    dec = columns.Float()
    object_name = columns.Text()
    radius = columns.Float()


def bootstrap_elasticsearch(container):
    delete_index(container.config.archive.ingestion.storage.parameters.index())
    mapping_path = os.path.join(
        os.path.dirname(antares.external.elasticsearch.__file__), "mapping.json"
    )
    with open(mapping_path) as f:
        mapping = json.load(f)
    create_index(
        container.config.archive.ingestion.storage.parameters.index(),
        mappings=mapping["mappings"],
    )


def generate_container_config_dict(datastore: str):
    return {
        "repositories": {
            "locus": {"datastore": datastore},
            "alert": {"datastore": datastore},
            "watch_list": {"datastore": datastore},
            "watch_object": {"datastore": datastore},
            "catalog": {"datastore": datastore},
        }
    }


def bootstrap_bigtable(container):
    wait_for_bigtable(
        project_id=container.config.repositories.locus.bigtable.parameters.project_id(),
        instance_id=container.config.repositories.locus.bigtable.parameters.instance_id(),
    )

    client = bigtable.Client(
        project=container.config.external.bigtable.parameters.project_id(), admin=True
    )
    if os.getenv("BIGTABLE_EMULATOR_HOST"):
        # see https://github.com/GoogleCloudPlatform/cloud-sdk-docker/issues/253#issuecomment-972247899
        # for details on why we do this
        client._emulator_channel = replace_emulator_channel_fn
    instance = client.instance(
        container.config.external.bigtable.parameters.instance_id()
    )
    bigtables_schemas_cfg = container.config.external.bigtable.schemas()
    for schema in BIGTABLE_SCHEMAS:
        schema = schema(bigtables_schemas_cfg)
        table = instance.table(schema.__table_name__)
        if not table.exists():
            log.info(
                f"Creating Bigtable table {schema.__table_name__},"
                f"column families={schema.column_families}"
            )
            table.create(column_families=schema.column_families)
    log.info("bootstrap_bigtable complete")


def bootstrap_cassandra(container):
    log.info("Bootstrapping Cassandra")
    # Create the keyspace and all necessary schema
    wait_for_cassandra()
    create_keyspace_simple(
        container.config.repositories.alert.cassandra.parameters.keyspace(),
        replication_factor=1,
    )
    cassandra_connection.setup(
        hosts=["cassandra"],
        default_keyspace=container.config.repositories.alert.cassandra.parameters.keyspace(),
    )
    os.environ["CQLENG_ALLOW_SCHEMA_MANAGEMENT"] = "1"  # per warning
    for schema in [*CASSANDRA_SCHEMAS, CassandraCatalogSchema]:
        sync_table(schema)


def bootstrap_mysql():
    log.info("Bootstrapping mysql")
    drop_main_db()
    engine = get_engine()
    sqlalchemy_utils.create_database(engine.url)
    metadata.create_all(bind=engine)
    session_factory = sessionmaker(engine)
    filter_repository = SqlAlchemyFilterRepository(session_factory)
    user_repository = SqlAlchemyUserRepository(session_factory)
    for user_ in users:
        user_repository.add(user_)
    for filter_ in filters:
        filter_repository.add(filter_)


def bootstrap():
    log.info("Bootstrapping services")

    container = Container()
    log.info("init container")
    container.config.from_yaml(
        "/home/delucchi/git/antares/antares/default.yaml", required=True
    )
    log.info("re-init container a")
    container.init_resources()
    log.info("re-init container b")
    # print("container", container)

    log.info("REAL bootstrapping")
    container = real_bootstrap.bootstrap(container)
    log.info("REAL bootstrapping -- cassandra")
    bootstrap_cassandra(container)
    bootstrap_bigtable(container)
    bootstrap_mysql()
    bootstrap_elasticsearch(container)
    return container
