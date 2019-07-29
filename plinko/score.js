let outputs = []

function normalize (data, featureCount) {
  const normalizedData = [...data]

  for (let i = 0; i < featureCount; i++) {
    const column = [...data].map(row => row[i])

    const min = Math.min(...column)
    const max = Math.max(...column)

    for (let j = 0; j < normalizedData.length; j++) {
      normalizedData[j][i] = (normalizedData[j][i] - min) / (max - min)
    }
  }

  return normalizedData
}

function shuffle (a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]]
  }
  return a
}

function splitDataSet (data, countToTest) {
  const shuffled = shuffle(data)
  const testSet = shuffled.slice(0, countToTest)
  const trainingSet = shuffled.slice(countToTest)

  return [testSet, trainingSet]
}

const sortBy = i => (a, b) => {
  if (a[i] < b[i]) return -1
  if (a[i] > b[i]) return 1
  return 0
}

function createPairs (obj) {
  return Object.entries(obj).map(([key, value]) => [key, value])
}

function distance (a, b) {
  const zipped = _.zip(a, b)
  return zipped.reduce((acc, [pointOne, pointTwo]) => {
    const sum = acc + Math.sqrt((pointOne - pointTwo) ** 2)
    return sum
  }, 0)
}

function calculatePointDeltas (testPoint) {
  const testInitial = testPoint.slice(0, testPoint.length - 1)
  return function ([a, b, c, bucket]) {
    return [distance([a, b, c], testInitial), bucket]
  }
}

function findMostCommonRecord (acc, [point, bucket]) {
  if (acc[bucket]) {
    acc[bucket] += 1
  } else {
    acc[bucket] = 1
  }

  return acc
}

function onScoreUpdate (dropPosition, bounciness, size, bucketLabel) {
  // Ran every time a balls drops into a bucket
  outputs = [...outputs, [dropPosition, bounciness, size, bucketLabel]]
}

function knn (data, point, k) {
  // calculate how far away from our testPoint each point in the training
  // data is
  const positions = data.map(calculatePointDeltas(point))

  // Sort the new points by the ones calculated to be most similar to
  // the point we are trying to test
  const sorted = positions.sort(sortBy(0))

  // Take the top K records from out sorted data
  const kRecords = sorted.slice(0, k)

  // find the most common record amongst the top K records
  const map = kRecords.reduce(findMostCommonRecord, {})
  // create array pairs of our top buckets and bucketCounts
  // sort those pairs by the bucket with the highest count
  const result = createPairs(map).sort(sortBy(1))

  // Choose the top record which is the bucket we are
  // predicting to have been the most similar to the test point
  const [bucket] = result[result.length - 1]

  // return the bucket we predicted the test point would
  // have fallen into
  return bucket
}

function runAnalysis () {
  const testSetSize = 100
  const k = 10

  _.range(0, 3).forEach(i => {
    const data = outputs.reduce((acc, row) => {
      const reducedData = [row[i], row.slice(-1)]
      return [...acc, reducedData]
    }, [])
    console.log(data)
    const [testSet, trainingSet] = splitDataSet(
      normalize(data, 1),
      testSetSize
    )

    const numCorrect = testSet.filter(
      // Compare how close our predicted bucket we received from our knn
      // algorithm actually was the the bucket in the point we are testing
      dataPoint => +(knn(trainingSet, dataPoint, k)) === +dataPoint.slice(-1)[0]
    ).length
    console.log(numCorrect)

    const accuracy = (numCorrect / testSetSize) * 100
    console.log(`accuracy: ${accuracy}%
    k value: ${k}`)
  })
}
