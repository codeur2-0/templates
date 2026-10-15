"""Source package of the project.

The package is organised by responsibility (one sub-package per concern), which keeps the
dependency graph shallow and makes each unit testable in isolation:

``data``          loading, validation contracts (Pandera), synthetic data generation
``preprocessing`` cleaning, scaling, encoding, feature pipelines
``features``      business-driven feature engineering
``models``        framework-specific models behind a common abstract interface
``training``      training loop, callbacks, metrics
``evaluation``    scoring, reports, error analysis
``inference``     batch / single-record prediction
``pipelines``     end-to-end orchestration
``schemas``       typed configuration (Pydantic) and API payloads
``utils``         logging, paths, IO, helpers
``visualization`` figures used by notebooks and reports
"""

__version__ = "1.0.0"
