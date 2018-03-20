import sys
import requests
import ast
import urllib2


mytoken = 'GxA0z2hlP1CpANphyllxamAjxOGlv3ci'

def get_company_info_api(text, token, category='organizations'):
    url = 'https://api.thomsonreuters.com/permid/search?q=%s' % urllib2.quote(text, safe='')
    access_token = token
    headers = {'X-AG-Access-Token' : access_token}
    try:
        print 'connecting to %s' % url
        response = requests.get(url, headers=headers)
    except Exception ,e:
        print 'Error in connect ' , e
        return
    print 'Status code: %s' % response.status_code
    if response.status_code == 200:
        print 'Results received.'
        response_dict = ast.literal_eval(response.content)
        return response_dict['result'][category]['entities'][0]

get_company_info_api('goldman',mytoken)
