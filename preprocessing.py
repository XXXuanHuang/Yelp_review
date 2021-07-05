import pyspark
import json

if __name__ == '__main__':

    sc_conf = pyspark.SparkConf() \
        .setAppName('task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '14g') \
        .set('spark.executor.memory', '7g') \
        .set("spark.driver.maxResultSize", "2g")

    sc = pyspark.SparkContext(conf=sc_conf)
    print('start collecting business dataset')
    b_rdd = sc.textFile("yelp_academic_dataset_business.json")
    b_rdd = b_rdd.map(lambda line:json.loads(line))
    def n_filter(x):
        temp = x["categories"]
        if temp == None:
            return False
        elif "Restaurants" in x["categories"]:
            return True

    # The businness part
    b_rdd = b_rdd.filter(lambda x: n_filter(x))
    b_rdd = b_rdd.map(lambda x:x["business_id"])
    b_list = b_rdd.collect()
    print('list of restaurant collected')

    # with open("b_filtered.json",'w') as f:
    #     temp = b_rdd.collect()
    #     for i in temp:
    #         f.write(json.dumps(i))
    #         f.write('\n')
    print('collecting review dataset')
    r_rdd = sc.textFile("yelp_academic_dataset_review.json")
    r_rdd = r_rdd.map(lambda line: json.loads(line))
    r_rdd = r_rdd.filter(lambda x: x["useful"]>=3 and len(x["text"].split(" "))>5)\
            .filter(lambda x:x["business_id"] in b_list)
    print('review dataset filtered')
    r_rdd = r_rdd.map(lambda x:{"user_id":x['user_id'],'business_id':x['business_id'],'stars':x['stars'],'text':x['text']})
    print('---')
    r_list = r_rdd.collect()

    # with open("r_filtered.json",'w') as f:
    #     for i in r_list:
    #         f.write(json.dumps(i))
    #         f.write('\n')
    print('dumping data')

    with open("b_r_filtered_20000.json", 'w') as f:
        for i in range(20000):
            f.write(json.dumps(r_list[i]))
            f.write('\n')

