import quandl


def get_quandl_data(symbollist, startdate, enddate):
    # startdate='2017-06-27', symbollist=['AAPL','TSLA']
    quandl.ApiConfig.api_key = "PmL3cEPezsZ_dAsyxrvX"
    print "downloading " + symbollist[0]
    df = quandl.get("WIKI/" + symbollist[0] + ".11", start_date=startdate, end_date=enddate)
    # "WIKI/AAPL.11"
    df.columns = [symbollist[0]]
    for stock in symbollist[1:]:
        print "downloading " + stock
        df[stock] = quandl.get("WIKI/" + stock + ".11", start_date=startdate, end_date=enddate)
    return df
