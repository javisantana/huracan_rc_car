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

    element.onmousedown = (event) => {
      this.tracking = true;
    }
    element.onmouseup = (event) => {
      this.tracking = false;
    }
    element.onmousemove = (event) => {
      this.realmouseX = ( event.clientX - widthHalf );
      this.realmouseY = ( event.clientY - heightHalf );
      this.mouseX = ( event.clientX - widthHalf ) / width;
      this.mouseY = ( event.clientY - heightHalf ) / height;
      if (this.tracking) {
        this.callback && this.callback(this.mouseX, this.mouseY);
        this.render();
      }
    };
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
      this.socket.onmessage = function(event) {
          console.log("event recieved", event);
      }
      this.checker();
  }

  checker() {
    setInterval( () => this.socket.send(`{ "cmd": "test"}`), 10000);
  }

  steering (d) {
    this.socket.send(JSON.stringify({ "cmd": "steering", "value": d}));
  }
}

var car = new Car();
car.start();

var input = new InputCanvas(document.getElementById('input'), 500, 500);
input.onMove((x, y) => {
  car.steering(x)
});
