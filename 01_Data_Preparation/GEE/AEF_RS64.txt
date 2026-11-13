var dataset = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
  .filterDate('2017-01-01', '2025-01-01');

var median = dataset.select('A57').median();
var median_with_proj = median.setDefaultProjection({
  crs: 'EPSG:4326',
  scale: 10 
});
var median_1km_resampled = median_with_proj
  .resample('bilinear')
  .reproject({
    crs: 'EPSG:4326',
    scale: 1000
  });
function makeGrid(xmin, ymin, xmax, ymax, dx, dy) {
  var tiles = [];
  for (var x = xmin; x < xmax; x += dx) {
    for (var y = ymin; y < ymax; y += dy) {
      var x2 = x + dx;
      var y2 = y + dy;
      if (x2 > 180) x2 = 180;
      if (y2 > 56) y2 = 84;
      
      var geometry = ee.Geometry.Rectangle([x, y, x2, y2], 'EPSG:4326', false);
      var name = 'Lon' + (x < 0 ? 'M' + Math.abs(x) : x) + 
                 '_Lat' + (y < 0 ? 'M' + Math.abs(y) : y);
      
      tiles.push({ region: geometry, name: name });
    }
  }
  return tiles;
}
var allTiles = makeGrid(-180, -56, 180, 84, 40, 40);

print('Tasks:', allTiles.length);

allTiles.forEach(function(tile) {
  Export.image.toDrive({
    image: median_1km_resampled.clip(tile.region), 
    description: 'GSE1km57_' + tile.name,
    folder: 'GSE1km_Global_57',    
    fileNamePrefix: 'GSE1km57_' + tile.name,
    region: tile.region,
    scale: 1000,
    crs: 'EPSG:4326',
    maxPixels: 1e13,
    skipEmptyTiles: true 
  });
});
var gridFC = ee.FeatureCollection(allTiles.map(function(t){ return ee.Feature(t.region); }));
Map.setCenter(0, 0, 1);
Map.addLayer(ee.Image().paint(gridFC, 0, 1), {palette: 'red'}, 'Export Grid');