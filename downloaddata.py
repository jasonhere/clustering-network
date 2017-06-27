import quandl


def get_quandl_data(symbollist, startdate, enddate):
    quandl.ApiConfig.api_key = "PmL3cEPezsZ_dAsyxrvX"
    df = quandl.get("WIKI/" + symbollist[0] + ".11", start_date=startdate, end_date=enddate)
    df.columns = [symbollist[0]]
    for stock in symbollist[1:]:
        df[stock] = quandl.get("WIKI/" + stock + ".11", start_date=startdate, end_date=enddate)
    return df
