import coreapi

def api_schema_generator():
    api_schema = coreapi.Document(
        title="Tweet Summarization Rest API",
        content={
            "APIs": {

                "summary": coreapi.Link(
                    url="/api/tweets/summary",
                    action="get",
                    description="Get a summary of tweets about the given query since the given date",
                    fields=[
                        coreapi.Field(
                            name="q",
                            required=True,
                            location="query",
                            description="The query (key words)"
                        ),
                        coreapi.Field(
                            name="since",
                            required=False,
                            location="query",
                            description="Collecte tweets created since this date (max. 7 days from today)"
                        )
                    ]
                ),
                "preprocess": coreapi.Link(
                    url="/api/tweets/process/",
                    action="post",
                    description="Preprocess the given list of json tweet objects",
                    fields=[
                        coreapi.Field(
                            name="tweets",
                            required=True,
                            location="body",
                            type="list",
                            description="List of json tweet objects"
                        )
                        
                    ]
                )
            }
        }
    )
    return api_schema