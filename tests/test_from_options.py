"""Tests for DAMWorkflow.from_options factory behavior."""

import pytest

from dam.workflow import workflow as workflow_module


DAMWorkflow = workflow_module.DAMWorkflow


class _FakeDataset:
    def __init__(self, payload=None):
        self.payload = payload

    @staticmethod
    def from_options(payload):
        return _FakeDataset(payload)


class TestDamWorkflowFromOptions:
    """Test DAM workflow object construction from options payloads."""

    def test_from_options_builds_workflow_and_processes(self, monkeypatch):
        init_calls = {}
        add_calls = []
        process_cfg = {
            "function": "my.process",
            "output": {"path": "/tmp", "file": "out.tif"},
            "pid": "p1",
            "a": 1,
        }
        options = {
            "input": {"path": "/tmp", "file": "in.tif"},
            "output": {"path": "/tmp", "file": "out.tif"},
            "options": {"intermediate_output": "Mem"},
            "processes": [process_cfg],
        }
        process_cfg_original = process_cfg.copy()

        monkeypatch.setattr(workflow_module, "Dataset", _FakeDataset)

        def _fake_init(self, input, output=None, options=None):
            init_calls["input"] = input
            init_calls["output"] = output
            init_calls["options"] = options

        def _fake_add_process(self, function, output=None, pid=None, **kwargs):
            add_calls.append(
                {
                    "function": function,
                    "output": output,
                    "pid": pid,
                    "kwargs": kwargs,
                }
            )

        monkeypatch.setattr(DAMWorkflow, "__init__", _fake_init)
        monkeypatch.setattr(DAMWorkflow, "add_process", _fake_add_process)

        wf = DAMWorkflow.from_options(options)

        assert isinstance(wf, DAMWorkflow)
        assert isinstance(init_calls["input"], _FakeDataset)
        assert isinstance(init_calls["output"], _FakeDataset)
        assert init_calls["options"] == {"intermediate_output": "Mem"}
        assert add_calls == [
            {
                "function": "my.process",
                "output": {"path": "/tmp", "file": "out.tif"},
                "pid": "p1",
                "kwargs": {"a": 1},
            }
        ]
        assert process_cfg == process_cfg_original

    def test_from_options_rejects_invalid_options_type(self):
        with pytest.raises(TypeError, match="'options' must be a mapping or an Options object"):
            DAMWorkflow.from_options(123)

    def test_from_options_requires_input(self):
        with pytest.raises(ValueError, match="Missing required 'input' in DAM workflow options"):
            DAMWorkflow.from_options({})

    def test_from_options_rejects_invalid_input_type(self, monkeypatch):
        monkeypatch.setattr(workflow_module, "Dataset", _FakeDataset)
        with pytest.raises(TypeError, match="'input' must be a Dataset or a mapping"):
            DAMWorkflow.from_options({"input": 123})

    def test_from_options_rejects_invalid_output_type(self, monkeypatch):
        monkeypatch.setattr(workflow_module, "Dataset", _FakeDataset)
        with pytest.raises(TypeError, match="'output' must be a Dataset, a mapping, or None"):
            DAMWorkflow.from_options({"input": {"x": 1}, "output": 123})

    def test_from_options_rejects_invalid_workflow_options_type(self, monkeypatch):
        monkeypatch.setattr(workflow_module, "Dataset", _FakeDataset)
        with pytest.raises(TypeError, match="'options' entry must be a mapping or None"):
            DAMWorkflow.from_options({"input": {"x": 1}, "options": "bad"})

    def test_from_options_rejects_invalid_processes_container_type(self, monkeypatch):
        monkeypatch.setattr(workflow_module, "Dataset", _FakeDataset)
        monkeypatch.setattr(DAMWorkflow, "__init__", lambda self, input, output=None, options=None: None)
        with pytest.raises(TypeError, match="'processes' must be a sequence of mappings"):
            DAMWorkflow.from_options({"input": {"x": 1}, "processes": {"function": "a"}})

    def test_from_options_rejects_non_mapping_process_items(self, monkeypatch):
        monkeypatch.setattr(workflow_module, "Dataset", _FakeDataset)
        monkeypatch.setattr(DAMWorkflow, "__init__", lambda self, input, output=None, options=None: None)
        with pytest.raises(TypeError, match="Each process entry must be a mapping"):
            DAMWorkflow.from_options({"input": {"x": 1}, "processes": ["bad"]})

    def test_from_options_requires_process_function(self, monkeypatch):
        monkeypatch.setattr(workflow_module, "Dataset", _FakeDataset)
        monkeypatch.setattr(DAMWorkflow, "__init__", lambda self, input, output=None, options=None: None)
        monkeypatch.setattr(DAMWorkflow, "add_process", lambda self, function, output=None, pid=None, **kwargs: None)
        with pytest.raises(ValueError, match="Each process entry must define a 'function'"):
            DAMWorkflow.from_options({"input": {"x": 1}, "processes": [{"pid": "p1"}]})
