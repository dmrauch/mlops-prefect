import prefect.context
import prefect.tasks
from typing import Any, Dict


def cache_within_flow_run(context: prefect.context.TaskRunContext,
                          arguments: Dict[str, Any]) -> str:
    '''
    Custom task hash function that only caches results within a single flow run

    This caching function can be applied to a task with ::

        @prefect.task(cache_key_fn=mlops_prefect.cache.cache_within_flow_run)

    '''
    return '{}-{}'.format(
        context.task_run.flow_run_id,
        prefect.tasks.task_input_hash(context, arguments))
