import datetime as dt
from prefect import flow, task

@task
def task1():
    print('task1')

@flow(name='hello-flow',
      flow_run_name=dt.datetime.now().strftime('%Y%m%d-%H%M%S'))
def hello_flow() -> None:
    '''
    This is a hello world flow
    '''
    print("This is a minimal flow - let's start!")
    task1()
    print("Finished the flow!")
