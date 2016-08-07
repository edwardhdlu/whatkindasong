// utility functions

function mapResponse(result, genres, accuracy) {
    if (result < 0) {
        $(".status").text("Sorry, I don't know what that sounds like :(");
    }
    else if (result == 0) {
        $(".status").text("Sorry, I haven't heard any songs like this :(");
    }
    else {
        $(".status").html("I am <strong>" + accuracy + "%</strong> sure this is a <strong>" + genres[result].replace("_", " ") + "</strong> song");;
    }
}

function toMatrix(arr) {
	matrix = [];
	for (var i = 0; i < arr.length; i++) {
		var row = arr[i];
		if (row.length > 0) {
			matrix.push(row.split(","));
		}
	}

	return matrix;
}

function matrixMultiply(m1, m2) {
	result = [];
	for (var i = 0; i < m1.length; i++) {
		var row = [];
		for (var j = 0; j < m2[0].length; j++) {
			sum = 0;
			for (var k = 0; k < m1[0].length; k++) {
				sum += m1[i][k] * m2[k][j];
			}
			row.push(sum);
		}
		result.push(row);
	}
	return result;
}

function matrixAdd(m1, m2) {
	result = [];
	for (var i = 0; i < m1.length; i++) {
		var row = [];
		for (var j = 0; j < m1[0].length; j++) {
			row.push(parseFloat(m1[i][j]) + parseFloat(m2[i][j]));
		}
		result.push(row);
	}
	return result;
}

function matrixOp(f, matrix) {
	var result = [];
	for (var i = 0; i < matrix.length; i++) {
		var row = [];
		for (var j = 0; j < matrix[0].length; j++) {
			row.push(f(matrix[i][j]));
		}
		result.push(row);
	}
	return result;
}

function tanh(matrix) {
	return matrixOp(Math.tanh, matrix);
}

function softmax(matrix) {
	var result = [];
	for (var i = 0; i < matrix.length; i++) {
		var row = [];
		for (var j = 0; j < matrix[0].length; j++) {
			var sum = 0;
			for (var k = 0; k < matrix.length; k++) {
				sum += Math.exp(matrix[k][j]);
			}
			var e_x = Math.exp(matrix[i][j]);
			row.push(e_x / sum);
		}
		result.push(row);
	}
	return result;
}

function transpose(matrix) {
	var rows = matrix.length;
	var cols = matrix[0].length;

	var result = [];
	for (var i = 0; i < cols; i++) {
		var row = [];
		for (var j = 0; j < rows; j++) {
			row.push(0);
		}
		result.push(row);
	}

	for (var i = 0; i < rows; i++) {
		for (var j = 0; j < cols; j++) {
			result[j][i] = matrix[i][j];
		}
	}
	return result;
}


// fetch data and handle user input

$(document).ready(function() {

    var words = words_str.split(",");
    var genres = genres_str.split(",");
    var len = genres.length - 1;

    $(".status").text("Fetching data...");

    var w_hidden, b_hidden;
    var w_hidden_2, b_hidden_2;
    var w_hidden_3, b_hidden_3;
    var w_output, b_output;

    const root = "https://raw.githubusercontent.com/edwardhdlu/whatkindasong/gh-pages/data/"

    var req1 = $.get(root + "w_hidden.csv", function(data) { w_hidden = data.split("\n"); });
    var req2 = $.get(root + "b_hidden.csv", function(data) { b_hidden = data.split("\n"); });
    var req3 = $.get(root + "w_hidden_2.csv", function(data) { w_hidden_2 = data.split("\n"); });
    var req4 = $.get(root + "b_hidden_2.csv", function(data) { b_hidden_2 = data.split("\n"); });
    var req5 = $.get(root + "w_hidden_3.csv", function(data) { w_hidden_3 = data.split("\n"); });
    var req6 = $.get(root + "b_hidden_3.csv", function(data) { b_hidden_3 = data.split("\n"); });
    var req7 = $.get(root + "w_output.csv", function(data) { w_output = data.split("\n"); });
    var req8 = $.get(root + "b_output.csv", function(data) { b_output = data.split("\n"); });

    $.when(req1, req2, req3, req4, req5, req6, req7, req8).done(function() {
    	var w1 = toMatrix(w_hidden);
    	var b1 = toMatrix(b_hidden);

    	var w2 = toMatrix(w_hidden_2);
    	var b2 = toMatrix(b_hidden_2);

    	var w3 = toMatrix(w_hidden_3);
    	var b3 = toMatrix(w_hidden_3);

    	var wo = toMatrix(w_output);
    	var bo = toMatrix(b_output);

    	$(".status").text("Ready! Please enter some lyrics");

    	$("#submit").click(function(e) {
    		$(".status").text("Processing...");
	        var counts = new Array(words.length).fill(0);

	        var input = $("#content").val();
	        var input_words = input.split(" ");

	        for (var i = 0; i < input_words.length; i++) {
	            var word = input_words[i].trim().replace(/[^a-zA-Z]/gi, "");
	            var index = words.indexOf(word);

	            if (index != -1) {
	                counts[index] = 1;
	            }
	        }

	        var x = [counts];
	        var h1 = tanh(matrixAdd(matrixMultiply(x, w1), b1));
	       	var h2 = tanh(matrixAdd(matrixMultiply(h1, w2), b2));
	       	var h3 = tanh(matrixAdd(matrixMultiply(h2, w3), b3));
	       	var y = softmax(transpose(matrixAdd(matrixMultiply(h3, wo), bo)));

	       	var t = transpose(y)[0];
	       	var result = t.indexOf(Math.max.apply(null, t));
	       	var accuracy = (t[result] * 100).toFixed(1);

	       	console.log(t);
	       	mapResponse(result, genres, accuracy);
	    });
    });
});