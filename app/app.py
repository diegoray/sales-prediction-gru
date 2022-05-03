import pandas as pd
from flask import Flask, render_template, request
from keras.models import load_model


app = Flask(__name__)


# load model from single file
gru_model = load_model('gru_model.h5')


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html')
    elif request.method == "POST":
        csv_file = request.files.get("file")
        df_pred = pd.read_csv(csv_file)

        # drop unnecessary column
        X = df_pred.drop_duplicates(subset=['barcode'])
        X.fillna(0, inplace=True)
        X.drop(['barcode'], axis=1, inplace=True)
        # X = df_pred.drop(['barcode'], axis=1, inplace=True)
        # reshape the predict dataset
        X_reshaped = X.values.reshape((X.shape[0], X.shape[1], 1))

        # predict the dataset
        model_pred = gru_model.predict(X_reshaped)

        # get back the barcode to pairing the prediction
        barcode_pred = df_pred[['barcode']]
        prediction = pd.DataFrame(barcode_pred['barcode'], columns=['barcode'])
        prediction['prediction_next_month'] = pd.DataFrame(model_pred)

        return prediction.to_html()

        # X = pd.read_csv(csv_file, parse_dates=['tgl'])
        # X = X.drop(['notxn', 'nonota', 'kodekategori', 'namabarang', 'hargajual', 'hargabeli', 'diskon', 'hargaafterdiskon', 'subtotal', 'kodeop', 'isbkp', 'kodecustomer',
        #            'iddistributor', 'idpromo', 'iddivisi', 'jenis', 'kategori', 'kodedepartemen', 'departemen', 'namaop', 'kodedivisibarang', 'divisibarang'], axis=1)
        # return render_template("index.html", data=X)


if __name__ == "__main__":
    app.run(debug=True)
