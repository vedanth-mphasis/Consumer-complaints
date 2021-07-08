#!/usr/bin/env python3

# This file implements the scoring service shell. You don't necessarily need to modify it for various
# algorithms. It starts nginx and gunicorn with the correct configurations and then simply waits until
# gunicorn exits.
#
# The flask server is specified to be the app object in wsgi.py
#
# We set the following parameters:
#
# Parameter                Environment Variable              Default Value
# ---------                --------------------              -------------
# number of workers        MODEL_SERVER_WORKERS              the number of CPU cores
# timeout                  MODEL_SERVER_TIMEOUT              60 seconds
from flask import request, Flask, Response
from io import StringIO
import pandas as pd
from main import engine


app = Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""

    status = 200
    return Response(response='Hey this is working', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as JSON, convert
    it to a pandas data frame for internal use and then convert the predictions back to JSON (which really
    just means one prediction per line, since there's a single column.
    """
    if request.content_type == 'text/csv':
        # Load data from CSV
        try:
            data = request.data.decode('utf-8')
            data_stream = StringIO(data)
            input_df = pd.read_csv(data_stream)
        except Exception as e:
            return Response(response='Could not create DataFrame from CSV, Please ensure format of data is valid\n{}'.format(e), status=500, mimetype='text/plain')
        try:
            print("3")
            output_df = engine(input_df)
            print("4:", output_df)
            return Response(response=output_df.to_csv(index=False), status=200, mimetype='text/csv', headers={"Content-disposition": "attachment; filename=results.csv"})
        except Exception as e:
            return Response(response='Could not process the input, please check the column name and data\n{}'.format(e), status=500, mimetype='text/plain')
    else:
        return Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080, debug = True)