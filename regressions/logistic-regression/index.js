require('@tensorflow/tfjs-node')
const LogisticRegression = require('./logistic-regression')
const plot = require('node-remote-plot')

const tf = require('@tensorflow/tfjs')
const loadCsv = require('../load-csv')

let { features, labels, testFeatures, testLabels } = loadCsv(
  '../data/cars.csv',
  {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
      passedemissions: value => value === 'TRUE' ? 1 : 0
    }
  }
)

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 10,
  batchSize: 10
})

plot({
  x: regression.costHistory
})

regression.train()
const test = regression.test(testFeatures, testLabels)
console.log(test)
