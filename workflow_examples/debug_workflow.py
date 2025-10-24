import argparse

from d3tools import Options
from d3tools.timestepping import get_date_from_str

from src.dam import DAMWorkflow

def parse_arguments():
    """
    Parse command line arguments for the workflow.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Run workflow with specified parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('workflow_json', type=str, help='JSON file of the workflow')
    parser.add_argument('-s', '--start', type=str, required=True, help='Start date to run the workflow [YYYY-MM-DD]')
    parser.add_argument('-e', '--end',   type=str, required=True, help='End date to run the workflow [YYYY-MM-DD]')
    
    args = parser.parse_args()

    return args

def main():
    args = parse_arguments()

    # load the options from the json file
    options = Options.load(args.workflow_json)

    # set the start and end date
    start_date = get_date_from_str(args.start) if args.start else None
    end_date   = get_date_from_str(args.end)   if args.end   else None

    # create the workflow
    wfs = [DAMWorkflow.from_options(wf) for wf in options.DAM_WORKFLOW] if isinstance(options.DAM_WORKFLOW, list) else [DAMWorkflow.from_options(options.DAM_WORKFLOW)]

    # run the computation
    for wf in wfs:
        wf.run([start_date, end_date])
        wf = None
    
if __name__ == '__main__':
    main()