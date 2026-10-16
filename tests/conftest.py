import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.orm import sessionmaker

import app.models  # noqa: F401  (registers tables)
from app.db.base import Base
from app.db.session import get_db, make_engine
from app.main import app
from app.models.alert import Alert
from app.models.prediction import Prediction


@pytest.fixture(scope="session")
def test_engine(tmp_path_factory):
    db_file = tmp_path_factory.mktemp("db") / "test.db"
    engine = make_engine(f"sqlite:///{db_file.as_posix()}")
    assert "test.db" in str(engine.url)  # never the real app.db
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def testing_session(test_engine):
    return sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)


@pytest.fixture(scope="session")
def client(testing_session):
    def override_get_db():
        db = testing_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:  # runs the lifespan, so the real model loads
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def clean_db(testing_session):
    with testing_session() as db:
        db.execute(delete(Alert))
        db.execute(delete(Prediction))
        db.commit()
    yield