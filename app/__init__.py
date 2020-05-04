from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask_migrate import Migrate

app = None
db = SQLAlchemy()
login_manager = LoginManager()
login_manager.session_protection = 'basic'
login_manager.login_view = 'auth.login'
mail = Mail()


def create_app(test_config=None):
    global db
    global login_manager
    global app

    app = Flask(__name__, instance_relative_config=True)
    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_pyfile('config_test.py', silent=True)

    db.init_app(app)
    login_manager.init_app(app)
    mail.init_app(app)

    from . import auth
    app.register_blueprint(auth.bp, url_prefix='/auth')
    from . import homepage
    app.register_blueprint(homepage.bp)
    from . import notification
    app.register_blueprint(notification.bp, url_prefix='/notification')
    from . import namelist
    app.register_blueprint(namelist.bp, url_prefix='/namelist')
    from . import housepage
    app.register_blueprint(housepage.bp, url_prefix='/house')
    from . import chat
    app.register_blueprint(chat.bp, url_prefix='/chat')
    from . import module_info
    app.register_blueprint(module_info.bp, url_prefix='/module')
    from . import searching
    app.register_blueprint(searching.bp)
    from . import training
    app.register_blueprint(training.bp)
    from app.models import db
    migrate = Migrate(app, db, render_as_batch=True)

    return app


if __name__ == '__main__':
    create_app()
