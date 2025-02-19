#!/usr/bin/env python3
"""
agent_functions.py

This module defines the metadata for the functions available to the Enhanced Meta Agent.
Each function is described with its name, description, and the parameters schema.
"""

AGENT_FUNCTIONS = [
    {
        "name": "create_agent_package",
        "description": "Spawn a new child agent with a specified name.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the new agent."
                },
                "base_path": {
                    "type": "string",
                    "description": "Optional base directory; default is F:/agents"
                }
            },
            "required": ["agent_name"]
        }
    },
    {
        "name": "write_text_file",
        "description": "Write plain text content to a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Full file path, including file name."
                },
                "content": {
                    "type": "string",
                    "description": "Text content to write."
                }
            },
            "required": ["filepath", "content"]
        }
    },
    {
        "name": "read_text_file",
        "description": "Read plain text content from a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "The file path to read."
                }
            },
            "required": ["filepath"]
        }
    },
    {
        "name": "run_agent_script",
        "description": "Execute the main script for an agent and capture output.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent to run."
                },
                "base_path": {
                    "type": "string",
                    "description": "Optional base directory; default is F:/agents"
                }
            },
            "required": ["agent_name"]
        }
    },
    {
        "name": "write_agent_tasks",
        "description": "Write tasks to a child agent's tasks.json and run that agent.",
        "parameters": {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent to receive tasks."
                },
                "tasks": {
                    "type": "array",
                    "description": "Array of tasks, each with a 'name' and 'prompt'.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "prompt": {"type": "string"}
                        },
                        "required": ["name", "prompt"]
                    }
                },
                "base_path": {
                    "type": "string",
                    "description": "Optional base directory; default is F:/agents"
                }
            },
            "required": ["agent_name", "tasks"]
        }
    },
    {
        "name": "list_directory_contents",
        "description": "List the files and folders in a given directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "Directory path"
                }
            },
            "required": ["directory_path"]
        }
    },
    {
        "name": "run_simulation_chamber",
        "description": (
            "Run a dynamic simulation chamber scenario that simulates live interactions among agents. "
            "Provide simulation_parameters with keys: scenario_description (string), agents (array of agent names), "
            "rounds (integer), and an optional analysis (boolean)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "simulation_parameters": {
                    "type": "object",
                    "description": (
                        "Parameters for the simulation chamber. "
                        "Expected keys: scenario_description, agents, rounds, and analysis (optional)."
                    ),
                    "properties": {
                        "scenario_description": {
                            "type": "string",
                            "description": "Description of the simulation scenario."
                        },
                        "agents": {
                            "type": "array",
                            "description": "List of agent names to simulate.",
                            "items": {"type": "string"}
                        },
                        "rounds": {
                            "type": "integer",
                            "description": "Number of conversation rounds to simulate."
                        },
                        "analysis": {
                            "type": "boolean",
                            "description": "Include detailed analysis in the report (optional)."
                        }
                    },
                    "required": ["scenario_description", "agents", "rounds"]
                }
            },
            "required": ["simulation_parameters"]
        }
    },
    {
        "name": "analyze_image",
        "description": "Analyze an image to answer a question about its content using DALL-E 3 vision capabilities.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file to be analyzed."
                },
                "question": {
                    "type": "string",
                    "description": "A question about the image (e.g., 'What is in this image?')."
                }
            },
            "required": ["image_path", "question"]
        }
    },
    {
        "name": "create_image",
        "description": "Generate an image based on a textual prompt using DALL-E 3.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text prompt to generate the image."
                },
                "output_path": {
                    "type": "string",
                    "description": "Optional file path to save the generated image."
                }
            },
            "required": ["prompt"]
        }
    }
]
