import antares.devkit as dk

from tests.fakes import build_filter_context
from superphot_plus_antares import example_module
from antares.devkit.filter.harness import RunnableFilter
from superphot_plus_antares.superphot_plus_v1 import SuperphotPlusSNClassification

from antares.domain.models import FilterContext


def test_greetings(int_alert_repo) -> None:
    """Verify the output of the `greetings` function"""
    # dk.init()
    locus = build_filter_context()
    print("locus", locus)
    filter = SuperphotPlusSNClassification()
    filter.setup()

    filter_locus = FilterContext.from_file("ANT2019ggddg.json")
    print("filter_locus", filter_locus)

    filter.run(locus)
    runnable_filter = RunnableFilter(SuperphotPlusSNClassification())
    returned_value = runnable_filter.run(locus)
    print("returned_value", returned_value)

    filter.run(filter_locus)
