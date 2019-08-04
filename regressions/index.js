require('@tensorflow/tfjs-node')
const LinearRegression = require('./linear-regression')
const plot = require('node-remote-plot')

const tf = require('@tensorflow/tfjs')
const loadCsv = require('./load-csv')

const { features, labels, testFeatures, testLabels } = loadCsv('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['mpg']
})

const linearRegression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 100,
  batchSize: 10
})

linearRegression.train()
const coefficientOfDetermination = linearRegression.test(testFeatures, testLabels)

plot({
  x: linearRegression.mseHistory.reverse(),
  xLabel: 'Iteration Number',
  yLabel: 'Mean Squared Error'
})

console.log(coefficientOfDetermination)
// console.log('update M is: ', linearRegression.weights.get(1, 0))
// console.log('update B is: ', linearRegression.weights.get(0, 0))
