let xvalues = [];
let yvalues = [];

let m, b;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function setup() {
  createCanvas(1000, 500);
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
}

function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  // y = mx + b;
  const ys = xs.mul(m).add(b);
  return ys;
}

function mousePressed() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);
  xvalues.push(x);
  yvalues.push(y);
}

function draw() {
  background(50);

  tf.tidy(() => {
    if (xvalues.length > 0) {
      const ys = tf.tensor1d(yvalues);
      optimizer.minimize(() => loss(predict(xvalues), ys));
    }
  });

 stroke(255,0,0);
  strokeWeight(8);
  for (let i = 0; i < xvalues.length; i++) {
    let px = map(xvalues[i], 0, 1, 0, width);
    let py = map(yvalues[i], 0, 1, height, 0);
    point(px, py);
  }


  const lineX = [0, 1];

  const ys = tf.tidy(() => predict(lineX));
  let lineY = ys.dataSync();
  ys.dispose();

  let x1 = map(lineX[0], 0, 1, 0, width);
  let x2 = map(lineX[1], 0, 1, 0, width);

  let y1 = map(lineY[0], 0, 1, height, 0);
  let y2 = map(lineY[1], 0, 1, height, 0);
  stroke(0,255,0);
  strokeWeight(4);
  line(x1, y1, x2, y2);


  console.log(tf.memory().numTensors);
  //noLoop();
}
