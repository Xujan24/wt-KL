import requests


if __name__ ==  "__main__":
    token_endpoint = 'https://icdaccessmanagement.who.int/connect/token'
    client_id = 'a50a0ba4-893b-4d91-b8b0-8d728ef58ad1_92465f89-0600-4aa1-a1bb-3118e29b1448'
    client_secret = 'cQzLSVn8b68BBMum7sGyvrDkQfW/NxTv1uUnC39CSrM='
    scope = 'icdapi_access'
    grant_type = 'client_credentials'

    # set data to post
    payload = {'client_id': client_id, 
            'client_secret': client_secret, 
            'scope': scope, 
            'grant_type': grant_type}
            
    # make request
    r = requests.post(token_endpoint, data=payload, verify=False).json()
    token = r['access_token']

    uri = 'https://id.who.int/icd/entity'

    # HTTP header fields to set
    headers = {'Authorization':  'Bearer '+token, 
            'Accept': 'application/json', 
            'Accept-Language': 'en',
        'API-Version': 'v2'}
            
    # make request           
    r = requests.get(uri, headers=headers, verify=False)

    # print the result
    print (r.text)			