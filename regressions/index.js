require('@tensorflow/tfjs-node')
const LinearRegression = require('./linear-regression')

const tf = require('@tensorflow/tfjs')
const loadCsv = require('./load-csv')

const { features, labels, testFeatures, testLabels } = loadCsv('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower'],
  labelColumns: ['mpg']
})

const linearRegression = new LinearRegression(features, labels, {
  learningRate: 0.0001,
  iterations: 100
})

linearRegression.features.print()
linearRegression.train()
const coefficientOfDetermination = linearRegression.test(testFeatures, testLabels)

console.log(coefficientOfDetermination)
// console.log('update M is: ', linearRegression.weights.get(1, 0))
// console.log('update B is: ', linearRegression.weights.get(0, 0))
