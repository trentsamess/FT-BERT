import os

import torch
import tornado.ioloop
import tornado.web

from model import model, predict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(os.path.join('/Users/kilDz/Downloads/', f"bert_weights.pth"), map_location=device.type))
model.eval()


class BertHandler(tornado.web.RequestHandler):
    def get(self):
        form = """<form method="post">
        <input type="text" name="sentence"/>
        <input type="submit"/>
        </form>"""
        self.write(form)

    def post(self):
        sentence = self.get_argument('sentence')
        with torch.no_grad():
            prediction = predict(sentence, 40)
        self.write(prediction)


def make_app():
    return tornado.web.Application([
        (r"/", BertHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(3000)
    tornado.ioloop.IOLoop.current().start()
