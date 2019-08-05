require('@tensorflow/tfjs-node')
const R = require('ramda')
const MultinominalLogisticRegression = require('./multinominal-logistic-regression')
const mnist = require('mnist-data')

const mnistData = mnist.training(0, 60000)
const removeLayer = x => x
const features = mnistData.images.values.map(img => R.chain(removeLayer, img))
const encodedLabels = mnistData.labels.values.map(label => {
  const row = Array(10).fill(0)
  row[label] = 1

  return row
})

const regression = new MultinominalLogisticRegression(features, encodedLabels, {
  learningRate: 0.1,
  iterations: 20,
  batchSize: 100
})

regression.train()

const mnistTest = mnist.testing(0, 1000)
const testFeatures = mnistTest.images.values.map(img => R.chain(removeLayer, img))
const testEncodedLabels = mnistTest.labels.values.map(label => {
  const row = Array(10).fill(0)
  row[label] = 1

  return row
})

const accuracy = regression.test(testFeatures, testEncodedLabels)
console.log(accuracy)

// const tf = require('@tensorflow/tfjs')
// const loadCsv = require('../load-csv')

// let { features, labels, testFeatures, testLabels } = loadCsv(
//   '../data/cars.csv',
//   {
//     dataColumns: ['horsepower', 'displacement', 'weight'],
//     labelColumns: ['mpg'],
//     shuffle: true,
//     splitTest: 50,
//     converters: {
//       mpg: value => {
//         const mpg = parseFloat(value)

//         if (mpg < 15) {
//           return [1, 0, 0]
//         } else if (mpg < 30) {
//           return [0, 1, 0]
//         }

//         return [0, 0, 1]
//       }
//     }
//   }
// )

// const removeLayer = x => x

// const regression = new MultinominalLogisticRegression(features, R.chain(removeLayer, labels), {
//   learningRate: 0.5,
//   iterations: 10,
//   batchSize: 10
// })

// regression.train()
// regression.predict([
//   [130, 307, 1.752]
// ]).print()
// const test = regression.test(testFeatures, R.chain(removeLayer, testLabels))
// console.log(test)
