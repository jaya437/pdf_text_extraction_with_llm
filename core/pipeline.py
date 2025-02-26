import logging
from typing import Annotated, TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from processors.text_extractor import extract_text_generic, extract_text_pypdf
from processors.text_comparison import compare_extraction
from core.models import PageState

logger = logging.getLogger("pdf_processor.pipeline")


# Define state for type checking with newer LangGraph
class PipelineState(TypedDict):
    state: PageState


def build_pipeline() -> StateGraph:
    """
    Build and compile the LangGraph pipeline with enhanced nodes.
    Pipeline flow:
      START -> ExtractTextLLM -> ExtractTextPyPDF -> CompareExtraction -> END

    """
    # Create a state graph with the PageState as the state type
    builder = StateGraph(PipelineState)

    # Each node function now receives the state dict and should return an updated state dict
    builder.add_node("ExtractTextLLM", lambda state: {"state": extract_text_generic(state["state"])})
    builder.add_node("ExtractTextPyPDF", lambda state: {"state": extract_text_pypdf(state["state"])})
    builder.add_node("CompareExtraction", lambda state: {"state": compare_extraction(state["state"])})

    # Set up the flow
    builder.add_edge("ExtractTextLLM", "ExtractTextPyPDF")
    builder.add_edge("ExtractTextPyPDF", "CompareExtraction")
    builder.add_edge("CompareExtraction", END)

    # Set the entry point
    builder.set_entry_point("ExtractTextLLM")

    # Compile the graph
    graph = builder.compile()

    logger.info("LangGraph pipeline built successfully with enhanced nodes.")
    return graph


def execute_pipeline(state: PageState) -> PageState:
    """
    Execute the pipeline on a single state object.
    This is a wrapper function that handles the conversion between the PageState
    object and the dictionary state expected by LangGraph

    Args:
        state: Initial PageState object

    Returns:
        Final PageState object after processing
    """
    pipeline = build_pipeline()

    # Convert to state dict for LangGraph 0.2.x
    state_dict = {"state": state}

    # Execute pipeline
    try:
        result = pipeline.invoke(state_dict)
        # Extract the final state from the result
        final_state = result["state"]
        logger.info(f"Pipeline execution completed for page {state.page_num}.")
        return final_state
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise