from flask import Flask
from os import environ
from flask_restx import Api
from flask_cors import CORS
from flask_wtf.csrf import CSRFProtect


from src import camSwagger





app = Flask(__name__, static_folder='templates')
app.config['SECRET_KEY'] = 'Zootech'
CORS(app, resources={r"/*"})
csrf = CSRFProtect()
csrf.init_app(app)

doc_root = environ.get('API_ROOT') if environ.get('API_ROOT') is not None else '/'
doc_prefix = doc_root + 'api' if doc_root == '/' else doc_root + '/api'


api = Api(app,
          title='Api Swagger - AI ZOOTECH',
          version='1.0',
          description='Modelo swagger flask-restx',
          doc=doc_root,
          prefix=doc_prefix,
          decorators=[csrf.exempt]
    )

api.add_namespace(camSwagger.api, path=camSwagger.path)


@app.route('/swagger')
def static_file():
    return app.send_static_file('api.html')


@app.route('/swagger/<json>')
def static_json(json):
    return app.send_static_file('swagger.json')


if __name__ == '__main__':
    SERVER_HOST = environ.get('SERVER_HOST', '0.0.0.0')
    app.run(host=SERVER_HOST, port=5500, debug=(environ.get('ENV') != 'PRODUCTION'), threaded=True)
