# Breast Cancer Planner Agent System

A hierarchical multi-agent system for breast cancer diagnosis and treatment planning, built with LangGraph. This system employs a master-slave agent architecture where a central PlannerAgent orchestrates specialized sub-agents to provide comprehensive, evidence-based treatment recommendations.

## Overview

The system is designed to assist in breast cancer clinical decision-making by:
- Analyzing patient information and extracting key clinical features (pathology type, staging, immunohistochemistry markers, lymph node status, etc.)
- Retrieving similar historical cases using RAG (Retrieval-Augmented Generation) technology
- Searching medical literature and clinical guidelines (NCCN, CSCO, etc.)
- Generating personalized, structured treatment plans based on evidence synthesis

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Example

```bash
# Run the example script
python core.py
```

## Architecture

### Master Agent
- **PlannerAgent**: The central planning agent responsible for:
  - Parsing patient information and extracting clinical features
  - Determining diagnosis and staging based on pathology and immunohistochemistry results
  - Identifying information gaps (missing tests, incomplete data)
  - Formulating query strategies and delegating tasks to sub-agents

### Sub-Agents
- **CaseRAGAgent**: Retrieves similar historical breast cancer cases from a vector database, focusing on cases with matching pathology types, staging, and immunohistochemistry profiles
- **LiteratureRAGAgent**: Searches medical literature, clinical guidelines, and evidence-based resources for treatment recommendations
- **SummaryAgent**: Synthesizes all information to generate comprehensive, structured treatment plans

## Key Features

- **Hierarchical Multi-Agent Architecture**: Extensible master-slave design allows for easy addition of new specialized agents
- **RAG-Enhanced Retrieval**: Vector database integration for efficient similarity search in both case histories and medical literature
- **Evidence-Based Recommendations**: Combines clinical guidelines, research evidence, and similar case outcomes
- **Structured Output**: Generates detailed treatment plans with clear recommendations and evidence levels
- **Workflow Orchestration**: Uses LangGraph for state management and workflow execution

## Technology Stack

- **LangGraph**: Workflow orchestration and state management
- **LangChain**: LLM integration and message handling
- **Vector Database**: RAG-based retrieval for cases and literature
- **Local LLM Support**: Compatible with local model deployments (e.g., Qwen2.5-7B-Instruct)

## Workflow

1. **Planning Phase**: PlannerAgent analyzes patient information and creates a query strategy
2. **Retrieval Phase**: CaseRAGAgent and LiteratureRAGAgent execute parallel searches
3. **Synthesis Phase**: SummaryAgent integrates all information and generates final recommendations

## Extensibility

The system is designed with extensibility in mind. New specialized agents can be easily integrated into the workflow to handle additional tasks such as:
- Risk assessment
- Drug interaction checking
- Follow-up planning
- Patient education content generation

