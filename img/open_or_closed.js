option = {
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      // Use axis to trigger tooltip
      type: 'shadow' // 'shadow' as default; can also be 'line' or 'shadow'
    }
  },
  legend: {},
  xAxis: {
    type: 'value'
  },
  yAxis: {
    type: 'category',
    data: ['Open', 'Closed', 'Open-noCompaction', 'closed-noCompaction']
  },
  series: [
    {
      name: 'generateRayFromCamera',
      type: 'bar',
      stack: 'total',
      label: {
        show: true
      },
      emphasis: {
        focus: 'series'
      },
      data: [1.024, 1.048, 1.070, 1.023]
    },
    {
      name: 'computeIntersections',
      type: 'bar',
      stack: 'total',
      label: {
        show: true
      },
      emphasis: {
        focus: 'series'
      },
      data: [24.103, 30.000, 40.320, 42.565]
    },
    {
      name: 'shadeBSDFMaterial',
      type: 'bar',
      stack: 'total',
      label: {
        show: true
      },
      emphasis: {
        focus: 'series'
      },
      data: [0.856, 1.128, 1.020, 1.365]
    },
  ]
};