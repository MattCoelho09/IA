from flask_restx import Resource
from flask_restx import Namespace
from flask_restx import reqparse
from flask_restx import fields
from flask import request
# from flask import send_file
import numpy 
import cv2
# from PIL import Image
# from datetime import datetime
from werkzeug.datastructures import FileStorage
import base64
import json

import plate_detect

path = '/placas'
api = Namespace('AI Zootech ', description='Mostra o resultado das cameras')
date_format = '%Y-%m-%d %H:%M:%S'


api_placa = Namespace('Plate Recognition', description='Reconhece os caracteres da placa')
# definição de modelo que será validado ao receber post
modelob64 = api_placa.model('Plate Decection', {
    'file': fields.String
})    

# parser = reqparse.RequestParser()
# parser.add_argument('line', type=str, required=True)
    
parser = reqparse.RequestParser()
parser.add_argument('file', location='files', type=FileStorage, required=True)
# # parser.add_argument('data', location='files', type=FileStorage, required=True)
# data_files = os.path.abspath(str(parser))
# print(data_files)
@api.route('')
class camURL(Resource):
    @api.expect(parser)
    @api.doc(responses={200: 'OK', 400: 'Bad Request', 500: 'Internal Server Error'})
    def post(self):
        try:
            #file_path = request.args['file']
            numpy_file = numpy.frombuffer(request.files['file'].read(), numpy.uint8)
            # # data_fileT = os.path.abspath(str(numpy_file))
            # # numpy_fileD = numpy.frombuffer(request.files['data'].read(), numpy.uint8)
            img = cv2.imdecode(numpy_file, cv2.IMREAD_COLOR)
            cv2.imwrite("/tmp/img.jpeg",img)
            retval, buffer = cv2.imencode('.jpg', img)
            y = base64.b64encode(buffer)
            z=str(y)
            z_new=z[2:-1]
            # PROCESSAR A IMAGEM
            def processar():
                import plate_detect
                from datetime import datetime

                plate_type,plate_text = plate_detect.run(source="/tmp/img.jpeg")
                hoje = datetime.today()
                horario = datetime.now()
                t = hoje.strftime("%d/%m/%Y")
                v = horario.strftime("%H:%M")
                m = str(v)
                k = str(t)
                return plate_type,plate_text,k,m
            p,t,d,j= processar()

            # datas = {
            #     'plate': p,
            #     'date':d,
            #     'base64_string':z_new
            # }
            # json = json.dumps(datas)
            
            # p = str(processar(self,img))
            
            # def detectar(self,numpy_fileD):
            #     import detect
            #     img = detect.run(self,numpy   _file,numpy_fileD)
            #     return img
            # im = detectar(self,numpy_fileD)
            # # # # # # # #
            
            return {   "message": "O nome da placa é:" + (t) +  "    Dia: " + (d) +  "     Horário: "  + (j)+  "    Modelo da Placa:"  + (p) }, 200
            #  A imagem em base64 é tal que:"  +  z_new  
        except Exception as err:
            return {"error": "Internal error: " + str(err)}, 500
    # @staticmethod
    # def get_cam(url):
    #     df = db.get_url(url)
    #     if df.empty:
    #         return []



  # json_data = json.loads(request.data)
        # data  = json_data['file']
        # if 'data:image/jpeg;base64,' in data:
        #     data = data.replace('data:image/jpeg;base64,', '')
        # imgdata = base64.b64decode(data)
        
        # numpy_file = numpy.frombuffer(imgdata, numpy.uint8)
        # img = cv2.imdecode(numpy_file, cv2.IMREAD_COLOR)

        # if img is None:
        #     response = make_response(jsonify({"error": "Image not valid."}), 400)
        #     response.headers["Content-Type"] = "application/json"
        #     return response
        # _, imencoded = cv2.imencode(".jpg", face)
        # img_as_txt = base64.b64encode(imencoded).decode('utf-8')

        # response = make_response(jsonify({"file": img_as_txt}), 200)
        # response.headers["Content-Type"] = "application/json"
        # return response