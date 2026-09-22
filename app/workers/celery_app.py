"""Celery application bootstrap.

TODO: Configure the broker, backend, task serialization, and worker settings.
"""

from celery import Celery

celery_app = Celery("docintel")
celery_app.config_from_object("app.workers.config", namespace="CELERY")
