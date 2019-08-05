require('@tensorflow/tfjs-node')
const LinearRegression = require('./linear-regression')
const plot = require('node-remote-plot')

const tf = require('@tensorflow/tfjs')
const loadCsv = require('../load-csv')

let { features, labels, testFeatures, testLabels } = loadCsv(
  '../data/cars.csv',
  {
    shuffle: true,
    splitTest: 20,
    dataColumns: ['horsepower', 'displacement', 'weight', 'cylinders'],
    labelColumns: ['mpg']
  }
)

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 5,
  batchSize: 10
})

regression.train()
const r2 = regression.test(testFeatures, testLabels)

plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error'
})

console.log('R2 is: ', r2)

regression.predict([
  [120, 380, 2, 4],
  [70, 90, 1, 6],
  [145, 380, 3, 8],
  [200, 345, 2.5, 8]
]).print()
