{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.7.10","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"none","dataSources":[{"sourceId":7099002,"sourceType":"datasetVersion","datasetId":1633303}],"dockerImageVersionId":30138,"isInternetEnabled":true,"language":"python","sourceType":"script","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"import pandas as pd\nimport numpy as np\n\ndf = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')\n\n\nfrom sklearn.preprocessing import StandardScaler\n\nscaler = StandardScaler()\ndf['f5_scale'] = scaler.fit_transform(df[['f5']])\nprint(df['f5_scale'].median()) #0.260619629559015","metadata":{"_uuid":"62ba023b-49e7-4e2d-aba1-ec32acec9430","_cell_guid":"19814400-1099-491d-beb2-3299008cbfe6","collapsed":false,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-05-20T13:19:33.735786Z","iopub.execute_input":"2024-05-20T13:19:33.736091Z","iopub.status.idle":"2024-05-20T13:19:33.751929Z","shell.execute_reply.started":"2024-05-20T13:19:33.736058Z","shell.execute_reply":"2024-05-20T13:19:33.750979Z"},"trusted":true},"execution_count":7,"outputs":[{"name":"stdout","text":"0.260619629559015\n","output_type":"stream"}]}]}