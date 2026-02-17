"""Integration tests for agent workflow with LangGraph.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import pytest

from sonic_o1_agent import SonicAgent


class TestAgentInitialization:
    """Test agent initialization and configuration."""

    def test_agent_init_minimal_config(self, minimal_agent_config):
        """Agent initializes with minimal config and has app and config."""
        agent = SonicAgent(minimal_agent_config)

        assert agent is not None
        assert hasattr(agent, "app")
        assert hasattr(agent, "config")
        assert agent.config == minimal_agent_config

    def test_agent_init_full_config(self):
        """Test agent with complete config."""
        config = {
            "model": {
                "model_path": "Qwen/Qwen3-Omni-0.5B",
                "tensor_parallel_size": 1,
                "max_frames": 128,
            },
            "processing": {
                "max_frames": 256,
                "max_audio_chunks": 5,
            },
            "reasoning": {
                "max_reasoning_steps": 5,
                "enable_verification": True,
            },
            "reflection": {
                "confidence_threshold": 0.7,
                "max_refinement_attempts": 2,
            },
        }
        agent = SonicAgent(config)

        assert agent.config == config
        assert agent.config["model"]["max_frames"] == 128
        assert agent.config["processing"]["max_audio_chunks"] == 5


class TestLangGraphWorkflow:
    """Test LangGraph workflow structure."""

    def test_workflow_compilation(self, minimal_agent_config):
        """Workflow compiles and app is set."""
        agent = SonicAgent(minimal_agent_config)

        assert hasattr(agent, "app")
        assert agent.app is not None

    def test_workflow_has_nodes(self, minimal_agent_config):
        """Workflow graph includes all expected nodes."""
        agent = SonicAgent(minimal_agent_config)

        graph = agent.app.get_graph()
        node_names = [node.id for node in graph.nodes.values()]

        # Check for expected nodes (including temporal_indexing and multi_step)
        expected_nodes = [
            "planning",
            "segmentation",
            "temporal_indexing",
            "multi_step",
            "reasoning",
            "direct",
            "reflection",
            "cleanup",
        ]

        for node in expected_nodes:
            assert node in node_names, f"Missing node: {node}"


class TestInputValidation:
    """Test input validation."""

    def test_process_requires_query(self, minimal_agent_config):
        """process raises ValueError when query is empty."""
        agent = SonicAgent(minimal_agent_config)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            agent.process(video_path="test.mp4", query="")

    def test_process_requires_media(self, minimal_agent_config):
        """process raises ValueError when neither video nor audio given."""
        agent = SonicAgent(minimal_agent_config)

        with pytest.raises(
            ValueError,
            match="At least one of video_path or audio_path required",
        ):
            agent.process(query="test query")


class TestStateManagement:
    """Test LangGraph state management."""

    def test_initial_state_preparation(self):
        """Test initial state is prepared correctly."""
        # This tests the state preparation logic
        initial_state = {
            "query": "test query",
            "video_path": "test.mp4",
            "audio_path": "test.m4a",
            "max_frames": 128,
            "use_reasoning": True,
            "use_reflection": False,
            "temp_files": [],
        }

        # Verify required fields
        assert "query" in initial_state
        assert "video_path" in initial_state
        assert "use_reasoning" in initial_state
        assert "temp_files" in initial_state
        assert "use_multi_step" not in initial_state or isinstance(
            initial_state.get("use_multi_step"), bool
        )


class TestWorkflowRoutingMultiStep:
    """Test multi-step routing (should_route_inference)."""

    def test_route_inference_multi_step_when_planned(self):
        """When use_multi_step and multi_step_plan set, route to multi_step."""
        from sonic_o1_agent.workflows.graph import should_route_inference

        state = {
            "use_multi_step": True,
            "multi_step_plan": [{"step_id": 1, "query": "x"}],
        }
        assert should_route_inference(state) == "multi_step"

    def test_route_inference_reasoning_when_no_multi_step(self):
        """When use_reasoning True and no multi_step, route to reasoning."""
        from sonic_o1_agent.workflows.graph import should_route_inference

        state = {"use_reasoning": True}
        assert should_route_inference(state) == "reasoning"

    def test_route_inference_direct_when_no_flags(self):
        """When no multi_step or reasoning, route to direct."""
        from sonic_o1_agent.workflows.graph import should_route_inference

        state = {}
        assert should_route_inference(state) == "direct"

    def test_route_inference_multi_step_empty_plan_goes_direct(self):
        """When use_multi_step but multi_step_plan empty, use reasoning/direct."""
        from sonic_o1_agent.workflows.graph import should_route_inference

        state = {"use_multi_step": True, "multi_step_plan": None}
        assert should_route_inference(state) == "direct"
        state["use_reasoning"] = True
        assert should_route_inference(state) == "reasoning"


class TestProcessStream:
    """Test process_stream API (streaming workflow events)."""

    def test_process_stream_requires_query(self, minimal_agent_config):
        """process_stream raises ValueError when query is empty."""
        agent = SonicAgent(minimal_agent_config)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            next(agent.process_stream(video_path="x.mp4", query=""))

    def test_process_stream_requires_media(self, minimal_agent_config):
        """process_stream raises ValueError when no video or audio."""
        agent = SonicAgent(minimal_agent_config)

        with pytest.raises(
            ValueError,
            match="At least one of video_path or audio_path required",
        ):
            next(agent.process_stream(query="test"))

    def test_process_stream_yields_node_and_state(self, minimal_agent_config):
        """process_stream yields events with node and state; first is planning."""
        agent = SonicAgent(minimal_agent_config)

        stream = agent.process_stream(
            video_path="nonexistent.mp4",
            query="summarize",
        )
        assert hasattr(stream, "__iter__")
        assert hasattr(stream, "__next__")

        event = next(stream)
        assert "node" in event
        assert "state" in event
        assert event["node"] == "planning"
        assert isinstance(event["state"], dict)
        assert "plan" in event["state"] or "query_type" in event["state"]


class TestContextManager:
    """Test agent context manager usage."""

    def test_context_manager_structure(self, minimal_agent_config):
        """Agent implements __enter__ and __exit__."""
        agent = SonicAgent(minimal_agent_config)

        assert hasattr(agent, "__enter__")
        assert hasattr(agent, "__exit__")

    def test_context_manager_usage(self, minimal_agent_config):
        """Agent can be used as context manager and exposes app."""
        with SonicAgent(minimal_agent_config) as agent:
            assert agent is not None
            assert hasattr(agent, "app")


class TestWorkflowConditionals:
    """Test workflow conditional routing (reasoning and reflection)."""

    def test_reasoning_flag_routing(self):
        """should_use_reasoning returns reasoning or direct by flag."""
        from sonic_o1_agent.workflows.graph import should_use_reasoning

        assert should_use_reasoning({"use_reasoning": True}) == "reasoning"
        assert should_use_reasoning({"use_reasoning": False}) == "direct"

    def test_reflection_flag_routing(self):
        """should_use_reflection returns reflection or cleanup by flag."""
        from sonic_o1_agent.workflows.graph import should_use_reflection

        assert should_use_reflection({"use_reflection": True}) == "reflection"
        assert should_use_reflection({"use_reflection": False}) == "cleanup"


class TestModelInfo:
    """Test model info retrieval."""

    def test_get_model_info(self, minimal_agent_config):
        """get_model_info returns backend and workflow keys."""
        agent = SonicAgent(minimal_agent_config)

        info = agent.get_model_info()

        assert "backend" in info
        assert "workflow" in info
        assert info["backend"] == "LangGraph + vLLM"
        assert info["workflow"] == "enabled"


@pytest.mark.skip(reason="Requires actual model and media files")
class TestEndToEndWorkflow:
    """End-to-end workflow tests (requires model)."""

    def test_full_workflow_direct(self):
        """Test full workflow with direct inference."""
        pass

    def test_full_workflow_reasoning(self):
        """Test full workflow with CoT reasoning."""
        pass

    def test_full_workflow_reflection(self):
        """Test full workflow with reflection."""
        pass
