import tornado.ioloop
import tornado.web
from transformers import BertForTokenClassification

from dataset import import_data, split_data, convert_dataframe_to_data, tags_and_tag_to_idx, tokenizer
from model import one_sentence_prediction_bert

path_to_dataset = '/home/andrei/Documents/ML/ner.csv'
path_to_model = '/home/andrei/Documents/ML/bert_uncased/'

data = import_data(path_to_dataset)
training, testing = split_data(data)
train_data = convert_dataframe_to_data(training)
test_data = convert_dataframe_to_data(testing)
_, tag_to_idx = tags_and_tag_to_idx(train_data, test_data)
tag_values = list(tag_to_idx.keys())

bert_model_loaded = BertForTokenClassification.from_pretrained(path_to_model)


class Ner(tornado.web.RequestHandler):

    def get(self):
        form = """<form method="post">
        <input type="text" name="sentence"/>
        <input type="submit"/>
        </form>"""
        self.write(form)

    def post(self):
        sentence = self.get_argument('sentence')
        prediction = one_sentence_prediction_bert(sentence, bert_model_loaded, tokenizer, tag_values)
        self.write(prediction)


application = tornado.web.Application([
    (r"/", Ner),
])

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
