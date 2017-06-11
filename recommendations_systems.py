import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# fetch data and format it

logger.info("get movie data with min rating 4")
data = fetch_movielens(min_rating=4.0)

# print training and testing data
logger.debug("print train data set")
logger.info(repr(data['train']))
logger.info(repr(data['test']))

# create model 
logger.info("Create LightFm model")
model = LightFM(loss = 'warp')
# train model
model.fit(data['train'], epochs=30, num_threads =2)

def sample_recomandation(model, data, user_ids):
	# number of user and movie in traning data
	n_users, n_items  = data['train'].shape
	logger.info("user ids : %s" %str(user_ids))

	# generate recommendations for each user we input
	for user_id in user_ids:
		# movie they already like
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		# movie our model predict that they will like
		scores = model.predict(user_id, np.arange(n_items))
		# rank them in order to most like to least
		top_items = data['item_labels'][np.argsort(-scores)]

		#  print  out the results
		print("User %s"% user_id )
		print("		Known positives:")

		# print top  3 liked 
		for x in known_positives[:3]:
			print("				%s" %x)

		print("Recommendations")

		for x in top_items[:3]:
			print("			%s" %x)


logger.info("result for 3 random user ids")

sample_recomandation(model, data, [3,25,450])


