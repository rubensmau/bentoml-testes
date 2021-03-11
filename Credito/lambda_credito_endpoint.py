import json,os
import boto3


#grab environment variables
endpoint_name = os.environ['ENDPOINT_NAME']
#endpoint_name  = "inference-pipeline-ep-2021-02-19-15-09-17"
# Acesso ao endpoint previsao    rf-scikit-2021-02-10-17-53-37-300

def lambda_handler(event, context):
    # TODO implement
    runtime = boto3.client('runtime.sagemaker')
    #print("Received event: " + json.dumps(event, indent=2))
    
    data = json.loads(json.dumps(event))
    #print(data)
    payload = data['data']
    #payload = [[0,0,0,"operario","sem conta",2,1,1,"muito_alta",2,1500,"Nao",2,6,"muito pobre","imovel","carro novo","proprio",2]]
    #payload = '0,0,0,operario,sem conta,2,1,1,muito_alta,2,1500,Nao,2,6,muito pobre,imovel,carro novo,proprio,2'             
   
    str1 = "" 
    for elem in payload[0]:
        str1 = str1 + str(elem) + ','
    payload1 = str1[:-1]  
    #payload2 = "0,0,0,operario,sem conta,2,1,1,muito_alta,2,1500,Nao,2,6,muito pobre,imovel,carro novo,proprio,2"
    print("payload   lido   =  ", payload1)
    #print("payload   manual =  ", payload2)
    # Send CSV text via InvokeEndpoint API
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='text/csv', Body=payload1)
    # Unpack response
    result = json.loads(response['Body'].read().decode())
    print(result)
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
