var fs = require('fs');

var content = fs.readFileSync('train.txt').toString().split('\n');
var lines = [];
for (var i = 0; i < content.length; i++) {
    if (content[i]) {
        lines.push(content[i]);
    }
}

var lineMap = {}
for (var i = 0; i < lines.length; i++) {
    var temp = lines[i].split(' ');
    var file = temp[0];
    var category = temp[1];
    if (!lineMap[category]) {
        lineMap[category] = []
    }
    s = file;
    s = s.slice(0, s.indexOf('.'));
    num = parseInt(s.slice(s.lastIndexOf('/') + 1))
    if (num <= 100) {
        lineMap[category].push(file);
    }
}

//console.log(lineMap);

newLines = [];
for (var category in lineMap) {
    if (lineMap.hasOwnProperty(category)) {
        for (var i = 0; i < lineMap[category].length; i++) {
            newLines.push(lineMap[category][i] + ' ' + category);
        }
    }
}

newContent = newLines.join('\n');
fs.writeFileSync('train_reduced.txt', newContent);