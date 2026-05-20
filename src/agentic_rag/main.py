import sys
import warnings
from .crew import build_langgraph_workflow

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")



def run():
    """
    Run the LangGraph workflow.
    """
    workflow = build_langgraph_workflow()
    inputs = {
        'query': 'What is the purpose of PDPA?'
    }
    result = workflow.invoke(inputs)
    print("LangGraph workflow result:", result)

def train():
    """
    Train the workflow for a given number of iterations.
    """
    pass

def replay():
    """
    Replay the workflow execution from a specific task.
    """
    pass

def test():
    """
    Test the workflow execution and returns the results.
    """
    pass
