# core
The core AI engine for Authormaton. This repository contains our proprietary agents and models responsible for autonomous research, verifiable factual synthesis, and comprehensive knowledge integration. It is the technological foundation for generating accurate, high-quality technical content.

## Project Structure

```
.
├── .coderabbit.yaml
├── .env.example
├── .gitignore
├── LICENSE
├── pytest.ini
├── README.md
├── requirements-dev.txt
├── requirements.txt
├── api
│   ├── __init__.py
│   ├── endpoints
│   │   ├── __init__.py
│   │   ├── internal.py
│   │   ├── upload.py
│   │   └── web_answering.py
│   ├── indexing_router.py
│   └── main.py
├── config
│   └── settings.py
├── data
│   └── uploads
├── experimentalCode
│   └── minimalchatbot.ipynb
├── models
│   ├── __init__.py
│   └── schemas.py
├── services
│   ├── __init__.py
│   ├── chunking_service.py
│   ├── embedding_service.py
│   ├── exceptions.py
│   ├── file_service.py
│   ├── logging_config.py
│   ├── parsing_service.py
│   ├── ranking_service.py
│   ├── synthesis_service.py
│   ├── vector_db_service.py
│   ├── web_fetch_service.py
│   ├── web_research_service.py
│   └── web_search_service.py
├── src
│   └── __init__.py
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── fixtures
    │   └── sample.pdf
    ├── test_chunking_embedding.py
    ├── test_embedding_batch.py
    ├── test_file_service.py
    ├── test_health.py
    ├── test_index_endpoint_e2e.py
    ├── test_internal_api.py
    ├── test_tavily_search.py
    ├── test_upload.py
    └── test_vector_db_service.py
```

## API Endpoints

### Health Check

- **GET `/health`**
  Returns a simple JSON response indicating the service status.
  Example response: `{"status": "ok"}`
