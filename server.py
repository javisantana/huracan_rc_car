#!/usr/bin/env python
#
# Copyright 2009 Facebook
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
"""Simplified chat demo for websockets.

Authentication, error handling, etc are left as an exercise for the reader :)
"""

import logging
import tornado.escape
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.websocket
import os.path
import os

from time import gmtime, strftime
import car

from tornado.options import define, options

define("port", default=8081, help="run on the given port", type=int)
define("image_interval", default="0.5", help="interval for image recoding")
define("replay", default='', help="folder to replay")

the_car = None


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", MainHandler),
            (r"/car", CarSocketHandler),
        ]
        settings = dict(
            cookie_secret="1233",
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            xsrf_cookies=True,
        )
        super(Application, self).__init__(handlers, **settings)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html", messages=CarSocketHandler.cache)


class CarSocketHandler(tornado.websocket.WebSocketHandler):
    waiters = set()
    cache = []
    cache_size = 200

    def get_compression_options(self):
        # Non-None enables compression with default options.
        return {}

    def open(self):
        CarSocketHandler.waiters.add(self)

    def on_close(self):
        CarSocketHandler.waiters.remove(self)

    @classmethod
    def send_updates(cls, chat):
        logging.info("sending message to %d waiters", len(cls.waiters))
        for waiter in cls.waiters:
            try:
                waiter.write_message(chat)
            except:
                logging.error("Error sending message", exc_info=True)

    def on_message(self, message):
        logging.info("got message %r", message)
        parsed = tornado.escape.json_decode(message)
        if parsed['cmd'] == "steering":
            the_car.steering(parsed['value'])
        if parsed['cmd'] == "throttle":
            the_car.throttle(parsed['value'])
        CarSocketHandler.send_updates(parsed)


def main():
    import base64, json
    global the_car
    tornado.options.parse_command_line()
    app = Application()
    app.listen(options.port)

    # initialize the car
    if options.replay:
        folder = options.replay
        print("replaying from %s" % options.replay)
    else:
        folder = "records/" + strftime("record_%a_%d_%b_%Y-%H_%M_%S", gmtime())
        os.mkdir(folder)
    the_car = car.Car()
    print("recoding images with interval %f" % float(options.image_interval))
    the_car.camera.start(folder, float(options.image_interval))
    def _send_image():
        image = the_car.camera.get_last_image()
        img_base64 = base64.b64encode(image)
        CarSocketHandler.send_updates(json.dumps({ 'cmd': 'camera_sensor', 'value': img_base64 }))
    tornado.ioloop.PeriodicCallback(_send_image, 500).start()
    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt as e:
        the_car.camera.stop()


if __name__ == "__main__":
    main()
