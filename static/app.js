console.log("working");



class InputCanvas {
  constructor (element, width, height) {

    this.ctx = element.getContext( '2d' );
    element.width = width;
    element.height = height;
    this.width = width;
    this.height = height;
    var widthHalf = width >> 1;
    var heightHalf = height >> 1;
    this.widthHalf = widthHalf;
    this.heightHalf = heightHalf;
    this.ctx.translate( widthHalf, heightHalf );

    this.mouseX = 0;
    this.mouseY = 0;
    this.realmouseX = 0;
    this.realmouseY = 0;
    this.tracking = false;


    element.ontouchstart = element.onmousedown = (event) => {
      this.tracking = true;
    }
    element.ontouchend = element.onmouseup = (event) => {
      this.tracking = false;
      this.resetThrottle();
    }
    function getXY(e) {
      if (e.touches) {
          var p = e.touches[0];
          return [p.pageX, p.pageY];
      } else {
          return [e.clientX, e.clientY];
      }
    }
    element.ontouchmove = element.onmousemove = (event) => {
      var xy = getXY(event);
      this.realmouseX = ( xy[0] - widthHalf );
      this.realmouseY = ( xy[1] - heightHalf );
      this.mouseX = ( xy[0] - widthHalf ) / width;
      this.mouseY = ( xy[1] - heightHalf ) / height;
      if (this.tracking) {
        this.callback && this.callback(this.mouseX, this.mouseY);
        this.render();
      }
    };
  }

  resetThrottle() {
    this.realmouseY = 0;
    this.callback && this.callback(this.mouseX, 0);
    this.render();
  }

  render() {
    let x = this.realmouseX, y = this.realmouseY;
    this.ctx.clearRect(-this.widthHalf, -this.heightHalf, this.width, this.height );
    this.ctx.beginPath();
    this.ctx.arc(x, y, 10, 0, 2 * Math.PI, true);
    this.ctx.stroke();
  }

  onMove (fn) {
    this.callback = fn;
  }

}

class Car {

  constructor () {
    this.socket = null;
  }

  start () {
      var url = "ws://" + location.host + "/car";
      this.socket = new WebSocket(url);
      this.socket.onmessage = (event) => {
          console.log("event recieved", event);
          var ev = JSON.parse(event.data)
          if (ev.cmd === 'camera_sensor') {
            this.onCameraSensor && this.onCameraSensor (ev.value)
          }
      }
      this.checker();
  }

  checker() {
    setInterval( () => this.socket.send(`{ "cmd": "test"}`), 10000);
  }

  steering (d) {
    this.socket.send(JSON.stringify({ "cmd": "steering", "value": d }));
  }

  throttle(d) {
    this.socket.send(JSON.stringify({ "cmd": "throttle", "value": d }));
  }
}

var car = new Car();
car.start();
const CAMERA_SIZE = 200;


var w = document.body.clientWidth;
var h = document.body.clientHeight - 200;

var input = new InputCanvas(document.getElementById('input'), w, h);
input.onMove((x, y) => {
  car.steering(x)
  car.throttle(y);
});

var camera_image = document.createElement('img')
var camera = document.getElementById('camera');
camera.appendChild(camera_image)
camera_image.height = CAMERA_SIZE

car.onCameraSensor = (img) => {
  camera_image.src = "data:image/jpg;base64," + img
}
