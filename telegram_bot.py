import os

import numpy as np
import telebot
from joblib import load
from sklearn.base import BaseEstimator
from sklearn.neighbors import BallTree

TOKEN = ''

bot = telebot.TeleBot(TOKEN)


def softmax(x):
    proba = np.exp(-x)
    return proba / sum(proba)


class NeighborSampler(BaseEstimator):
    def __init__(self, k=5, temperature=1.0):
        self.k = k
        self.temperature = temperature
    def fit(self, X, y):
        self.tree_ = BallTree(X)
        self.y_ = np.array(y)
    def predict(self, X, random_state=None):
        distances, indices = self.tree_.query(X, return_distance=True, k=self.k)
        result = []
        for distance, index in zip(distances, indices):
            result.append(np.random.choice(index, p=softmax(distance * self.temperature)))
        return self.y_[result]


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, 'Привет. Давай поговорим!')


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.send_message(message.from_user.id, pipe.predict([message.text.lower()])[0])


if __name__ == "__main__":
    pipe = load('pipe.joblib')
    print("Start polling")
    bot.infinity_polling(timeout=10, long_polling_timeout=5)