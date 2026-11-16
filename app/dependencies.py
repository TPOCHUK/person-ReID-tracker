from fastapi import Request

from app.cv.pipeline import CVPipeline
from app.service.system import SystemService


def get_pipeline(request: Request) -> CVPipeline:
    return request.app.state.pipeline


def get_system_service(request: Request) -> SystemService:
    return request.app.state.system_service
