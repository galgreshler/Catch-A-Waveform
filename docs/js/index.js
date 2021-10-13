var sr_files;
var sr_files_8;
var inpainting_files;
var denoising_files;
var denoising_urls;
var speeches_files;
var instruments_files;
var instruments_names;
var ambient_files;
var limitations_files;
var music_files;
var music_names;
var music_artists;
var music_links;
var music_offsets;
var number_of_sr_files = 0;
var number_of_sr_8_files = 0;
var number_of_inpainting_files = 0;
var number_of_instruments_files = 0;
var number_of_speeches_files = 0;
var number_of_music_files = 0;
var number_of_ambient_files = 0;
var number_of_limitations_files = 0;
// var audio_base_url = 'https://catch-a-waveform.s3-us-west-2.amazonaws.com/';
var audio_base_url = '';

// load audio files and create tables
function load_instruments_files() {
    var is_small_screen = $('#instruments_table').parent().css('display') == 'none';
    for (var j = 0; j < 7; j++) {
        var idx = j + number_of_instruments_files;
        if (idx == instruments_files.length) {
            $("#add_instruments_button").hide()
            break
        }
        if (!is_small_screen) {
            tr = '<tr>';
            tr += '<th style="text-align: center"><img src="graphics/unconditional/instruments/' + instruments_names[idx] + '.png" height="100"/></th>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/instruments/' + instruments_files[idx] + '_real.ogg" controls></audio>' +
                '</td>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/instruments/' + instruments_files[idx] + '_fake1.ogg" controls></audio>' +
                '</td>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/instruments/' + instruments_files[idx] + '_fake2.ogg" controls></audio>' +
                '</td>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/instruments/' + instruments_files[idx] + '_fake3.ogg" controls></audio>' +
                '</td>';
            tr += '</tr>';
            $('#instruments_table').append(tr);
        } else {
            tr = '<tr>';
            tr += '<th style="text-align: center" rowspan="4" valign="top"><img src="graphics/unconditional/instruments/' + instruments_names[idx] + '.png" height="100"/></th>';
            tr += '<th>Input (Real)</th></tr>';
            tr2 = '<tr><td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/instruments/' + instruments_files[idx] + '_real.ogg" controls></audio>' +
                '</td></tr>';
            tr3 = '<tr><th>Fakes</th></tr>';
            tr4 = '<tr><td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/instruments/' + instruments_files[idx] + '_fake1.ogg" controls></audio>';
            tr4 += '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/instruments/' + instruments_files[idx] + '_fake2.ogg" controls></audio>';
            tr4 += '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/instruments/' + instruments_files[idx] + '_fake3.ogg" controls></audio>';
            tr4 += '</td></tr>';
            $('#instruments_table_small').append(tr);
            $('#instruments_table_small').append(tr2);
            $('#instruments_table_small').append(tr3);
            $('#instruments_table_small').append(tr4);
        }
    }
    number_of_instruments_files += j;
    if (number_of_instruments_files >= instruments_files.length) {
        $("#add_instruments_button").hide()
    }
}

function load_speeches_files() {
    var is_small_screen = $('#speeches_table').parent().css('display') == 'none';
    for (var j = 0; j < 5; j++) {
        var idx = j + number_of_speeches_files;
        if (idx == speeches_files.length) {
            $("#add_speeches_button").hide()
            break
        }
        var name;
        if (speeches_files[idx].includes('trump')) {
            name = 'Trump';
        } else if (speeches_files[idx].includes('RAP') || speeches_files[idx].includes('Acapella') || speeches_files[idx].includes('LoseYourself') || speeches_files[idx].includes('FastLife')) {
            name = 'Rap';
        } else if (speeches_files[idx].includes('bho')) {
            name = 'Obama';
        }
        if (!is_small_screen) {
            tr = '<tr>';
            tr += '<th style="text-align: center"><img src="graphics/unconditional/speeches/' + name + '.png" height="100"/></th>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/speeches/' + speeches_files[idx] + '_real.ogg" controls></audio>' +
                '</td>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/speeches/' + speeches_files[idx] + '_fake1.ogg" controls></audio>' +
                '</td>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/speeches/' + speeches_files[idx] + '_fake2.ogg" controls></audio>' +
                '</td>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/speeches/' + speeches_files[idx] + '_fake3.ogg" controls></audio>' +
                '</td>';

            tr += '</tr>';
            $('#speeches_table').append(tr);
        } else {
            tr = '<tr><th style="text-align: center" rowspan="4"><img src="graphics/unconditional/speeches/' + name + '.png" height="100"/></th>';
            tr += '<th>Input (Real)</th></tr>';
            tr2 = '<tr><td><audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/speeches/' + speeches_files[idx] + '_real.ogg" controls></audio></td></tr>';
            tr3 = '<tr><th>Fakes</th></tr>';
            tr4 = '<tr><td><audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/speeches/' + speeches_files[idx] + '_fake1.ogg" controls></audio>';
            tr4 += '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/speeches/' + speeches_files[idx] + '_fake2.ogg" controls></audio>';
            tr4 += '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/speeches/' + speeches_files[idx] + '_fake3.ogg" controls></audio></td></tr>'
            $('#speeches_table_small').append(tr);
            $('#speeches_table_small').append(tr2);
            $('#speeches_table_small').append(tr3);
            $('#speeches_table_small').append(tr4);
        }
    }
    number_of_speeches_files += j;
    if (number_of_speeches_files >= speeches_files.length) {
        $("#add_speeches_button").hide()
    }
}

function load_ambient_files() {
    var is_small_screen = $('#ambient_table').parent().css('display') == 'none';
    for (var j = 0; j < 5; j++) {
        var idx = j + number_of_ambient_files;
        if (idx == ambient_files.length) {
            $("#add_ambient_button").hide()
            break
        }
        var name;
        if (ambient_files[idx].includes('rain')) {
            name = 'Thunderstorm';
        } else if (ambient_files[idx].includes('Applause')) {
            name = 'Audience Applause';
        } else if (ambient_files[idx].includes('Crowd')) {
            name = 'Crowd Cheering';
        }
        if (!is_small_screen) {
            tr = '<tr>';
            tr += '<th style="text-align: center; width: 100px">' + name + '</th>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/ambient/' + ambient_files[idx] + '_real.ogg" controls></audio>' +
                '</td>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/ambient/' + ambient_files[idx] + '_fake1.ogg" controls></audio>' +
                '</td>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/ambient/' + ambient_files[idx] + '_fake2.ogg" controls></audio>' +
                '</td>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/ambient/' + ambient_files[idx] + '_fake3.ogg" controls></audio>' +
                '</td>';

            tr += '</tr>';
            $('#ambient_table').append(tr);
        } else {
            tr = '<tr><th style="text-align: center; width: 100px" rowspan="4">' + name + '</th><th>Input (Real)</th></tr>';
            tr2 = '<tr><td><audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/ambient/' + ambient_files[idx] + '_real.ogg" controls></audio></td></tr>';
            tr3 = '<tr><th>Fakes</th></tr>';
            tr4 = '<tr><td><audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/ambient/' + ambient_files[idx] + '_fake1.ogg" controls></audio>';
            tr4 += '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/ambient/' + ambient_files[idx] + '_fake2.ogg" controls></audio>';
            tr4 += '<audio class="short-audio" src="' + audio_base_url + 'audio/unconditional/ambient/' + ambient_files[idx] + '_fake3.ogg" controls></audio></td></tr>';
            $('#ambient_table_small').append(tr);
            $('#ambient_table_small').append(tr2);
            $('#ambient_table_small').append(tr3);
            $('#ambient_table_small').append(tr4);
        }
    }
    number_of_ambient_files += j;
}

function create_rf_table() {
    var is_small_screen = $('#rf_table').parent().css('display') == 'none';
    var lengths = [20, 45];
    for (var i = 0; i < 2; i++) {
        if (!is_small_screen) {
            tr = '<tr>';
            tr += '<td style="text-align: center; width: 100px">' + lengths[i] + '</td>';
            tr += '<td style="text-align: center"><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_real.ogg" controls></audio></td>';
            tr += '<td style="text-align: center"><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_small1_fake.ogg" controls></audio><br><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_small2_fake.ogg" controls></audio></td>';
            tr += '<td style="text-align: center"><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_normal1_fake.ogg" controls></audio><br><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_normal2_fake.ogg" controls></audio></td>';
            tr += '<td style="text-align: center"><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_large1_fake.ogg" controls></audio><br><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_large2_fake.ogg" controls></audio></td>';
            tr += '</tr>';
            $("#rf_table").append(tr);
        } else {
            tr = '<tr><th style="text-align: center; width: 100px" rowspan="8">' + lengths[i] + ' [sec] training length</th><th>Input (Real)</th></tr>';
            tr2 = '<tr><td><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_real.ogg" controls></audio></td></tr>';
            tr3 = '<tr><th>Small Receptive Field<br>2[sec]-60[ms]</th></tr>';
            tr4 = '<tr><td style="text-align: center"><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_small1_fake.ogg" controls></audio><br><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_small2_fake.ogg" controls></audio></td></tr>';
            tr5 = '<tr><th>Normal Receptive Field<br>4[sec]-120[ms]</th></tr>';
            tr6 = '<tr><td style="text-align: center"><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_normal1_fake.ogg" controls></audio><br><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_normal2_fake.ogg" controls></audio></td></tr>';
            tr7 = '<tr><th>Small Receptive Field<br>2[sec]-60[ms]</th></tr>';
            tr8 = '<tr><td style="text-align: center"><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_large1_fake.ogg" controls></audio><br><audio class="short-audio" src="' + audio_base_url + 'audio/rf/' + i + '_trump_large2_fake.ogg" controls></audio></td></tr>';
            $("#rf_table_small").append(tr);
            $("#rf_table_small").append(tr2);
            $("#rf_table_small").append(tr3);
            $("#rf_table_small").append(tr4);
            $("#rf_table_small").append(tr5);
            $("#rf_table_small").append(tr6);
            $("#rf_table_small").append(tr7);
            $("#rf_table_small").append(tr8);
        }
    }
}

function load_music_files() {
    var is_small_screen = $('#music_table').parent().css('display') == 'none';
    for (var j = 0; j < 5; j++) {
        var idx = j + number_of_music_files;
        if (idx == music_files.length) {
            $("#add_music_button").hide()
            break
        }
        if (!is_small_screen) {
            tr = '<tr>';
            // tr+='<th style="text-align: center"><a href="' + music_links[idx] + '?autoplay=1&t=0m' + music_offsets[idx] + 's" target="_blank">' + music_names[idx] + '</a></th>';
            tr += '<th style="text-align: center; width: 175px">' + music_names[idx] + '</th>';
            tr += '<th style="text-align: center; width: 175px">' + music_artists[idx] + '</th>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/music/' + music_files[idx] + '_real.ogg" controls></audio>' +
                '</td>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/music/' + music_files[idx] + '_fake1.wav" controls></audio>' +
                '</td>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/music/' + music_files[idx] + '_fake2.wav" controls></audio>' +
                '</td>';
            tr += '</tr>';
            $('#music_table').append(tr);
        } else {
            tr = '<tr><th style="text-align: center; width: 100px" rowspan="4">' + music_names[idx] + '<br><br>' + music_artists[idx] + '</th><th>Input (Real)</th></tr>';
            tr2 = '<tr><td><audio class="short-audio" src="' + audio_base_url + 'audio/music/' + music_files[idx] + '_real.ogg" controls></audio</td></tr>';
            tr3 = '<tr><th>Fakes</th></tr>';
            tr4 = '<tr><th><audio class="short-audio" src="' + audio_base_url + 'audio/music/' + music_files[idx] + '_fake1.wav" controls></audio><audio class="short-audio" src="' + audio_base_url + 'audio/music/' + music_files[idx] + '_fake2.wav" controls></audio></th></tr>';
            $('#music_table_small').append(tr);
            $('#music_table_small').append(tr2);
            $('#music_table_small').append(tr3);
            $('#music_table_small').append(tr4);
        }
    }
    number_of_music_files += j;
    if (number_of_music_files >= music_files.length) {
        $("#add_music_button").hide()
    }
}

function load_sr_files() {
    var is_small_screen = $('#sr_table').parent().css('display') == 'none';
    for (var j = 0; j < 3; j++) {
        var idx = j + number_of_sr_files;
        if (idx >= sr_files.length) {
            $("#add_sr_button").hide();
            break
        }
        if (!is_small_screen) {
            console.log('here')
            tr = '<tr><th colSpan="3" style="background-color: lightgrey; background-clip: padding-box">' + sr_files[idx] + '</th></tr>';

            tr2 = '<tr><td style="text-align: center">Input (Low Resolution)<div className="imgbox"><img className="center-fit" src="graphics/sr/' + sr_files[idx] + '_lr.png" width="300"></div>';
            tr2 += '<audio src="' + audio_base_url + 'audio/sr/' + sr_files[idx] + '_lr.ogg" controls></audio></td>';
            tr2 += '<td style="text-align: center">GT (High-Resolution)<div className="imgbox"><img className="center-fit" src="graphics/sr/' + sr_files[idx] + '_gt.png" width="300"></div>';
            tr2 += '<audio src="' + audio_base_url + 'audio/sr/' + sr_files[idx] + '_gt.ogg" controls></audio></td>';
            tr2 += '<td style="text-align: center">Our Training Signal<div className="imgbox"><img className="center-fit" src="graphics/sr/' + sr_files[idx] + '_training.png" width="300"></div>';
            tr2 += '<audio src="' + audio_base_url + 'audio/sr/' + sr_files[idx] + '_training.ogg" controls></audio></td></tr>';

            tr3 = '<tr><td style="text-align: center">TFiLM (single)<div className="imgbox"><img className="center-fit" src="graphics/sr/' + sr_files[idx] + '_pr_single.png" width="300"></div>';
            tr3 += '<audio src="' + audio_base_url + 'audio/sr/' + sr_files[idx] + '_pr_single.ogg" controls></audio></td>';
            tr3 += '<td style="text-align: center">TFiLM (multi)<div className="imgbox"><img className="center-fit" src="graphics/sr/' + sr_files[idx] + '_pr_multi.png" width="300"></div>';
            tr3 += '<audio src="' + audio_base_url + 'audio/sr/' + sr_files[idx] + '_pr_multi.ogg" controls></audio></td>';
            tr3 += '<td style="text-align: center">Ours<div className="imgbox"><img className="center-fit" src="graphics/sr/' + sr_files[idx] + '_ours.png" width="300"></div>';
            tr3 += '<audio src="' + audio_base_url + 'audio/sr/' + sr_files[idx] + '_ours.ogg" controls></audio></td></tr>';
            $('#sr_table').append(tr);
            $('#sr_table').append(tr2);
            $('#sr_table').append(tr3);
        } else {
            tr = '<tr><th colSpan="3" style="background-color: lightgrey; background-clip: padding-box">' + sr_files[idx] + '</th></tr>';
            tr2 = '<tr><td style="text-align: center; background-color: white">Input (Low Resolution)<div className="imgbox"><img className="center-fit" src="graphics/sr/' + sr_files[idx] + '_lr.png" width="300"></div>';
            tr2 += '<audio src="' + audio_base_url + 'audio/sr/' + sr_files[idx] + '_lr.ogg" controls></audio></td></tr>';
            tr2 += '<tr><td style="text-align: center; background-color: white">GT (High-Resolution)<div className="imgbox"><img className="center-fit" src="graphics/sr/' + sr_files[idx] + '_gt.png" width="300"></div>';
            tr2 += '<audio src="' + audio_base_url + 'audio/sr/' + sr_files[idx] + '_gt.ogg" controls></audio></td></tr>';
            tr2 += '<tr><td style="text-align: center; background-color: white">Our Training Signal<div className="imgbox"><img className="center-fit" src="graphics/sr/' + sr_files[idx] + '_training.png" width="300"></div>';
            tr2 += '<audio src="' + audio_base_url + 'audio/sr/' + sr_files[idx] + '_training.ogg" controls></audio></td></tr>';
            tr3 = '<tr><td style="text-align: center; background-color: white">TFiLM (single)<div className="imgbox"><img className="center-fit" src="graphics/sr/' + sr_files[idx] + '_pr_single.png" width="300"></div>';
            tr3 += '<audio src="' + audio_base_url + 'audio/sr/' + sr_files[idx] + '_pr_single.ogg" controls></audio></td>';
            tr3 += '<tr><td style="text-align: center; background-color: white">TFiLM (multi)<div className="imgbox"><img className="center-fit" src="graphics/sr/' + sr_files[idx] + '_pr_multi.png" width="300"></div>';
            tr3 += '<audio src="' + audio_base_url + 'audio/sr/' + sr_files[idx] + '_pr_multi.ogg" controls></audio></td></tr>';
            tr3 += '<tr><td style="text-align: center; background-color: white">Ours<div className="imgbox"><img className="center-fit" src="graphics/sr/' + sr_files[idx] + '_ours.png" width="300"></div>';
            tr3 += '<audio src="' + audio_base_url + 'audio/sr/' + sr_files[idx] + '_ours.ogg" controls></audio></td></tr>';
            $('#sr_table_small').append(tr);
            $('#sr_table_small').append(tr2);
            $('#sr_table_small').append(tr3);

        }
    }
    number_of_sr_files += j;
    if (number_of_sr_files >= sr_files.length) {
        $("#add_sr_button").hide()
    }
}

function load_sr_files_8() {
    var is_small_screen = $('#sr_table_8_to_16').parent().css('display') == 'none';
    for (var j = 0; j < 3; j++) {
        var idx = j + number_of_sr_8_files;
        if (idx >= sr_files_8.length) {
            $("#add_sr_button_8_to_16").hide();
            break
        }
        if (!is_small_screen) {
            tr = '<tr><th colSpan="4" style="background-color: lightgrey; background-clip: padding-box">' + sr_files_8[idx] + '</th></tr>';
            tr2 = '<tr><td style="text-align: center">Input (Low Resolution)<div className="imgbox"><img className="center-fit" src="graphics/sr_8_to_16/' + sr_files_8[idx] + '_lr.png" width="300"></div>';
            tr2 += '<audio src="' + audio_base_url + 'audio/sr_8_to_16/' + sr_files_8[idx] + '_lr.ogg" controls></audio></td>';
            tr2 += '<td style="text-align: center">GT (High-Resolution)<div className="imgbox"><img className="center-fit" src="graphics/sr_8_to_16/' + sr_files_8[idx] + '_gt.png" width="300"></div>'
            tr2 += '<audio src="' + audio_base_url + 'audio/sr_8_to_16/' + sr_files_8[idx] + '_gt.ogg" controls></audio></td></tr>';
            tr3 = '<tr><td style="text-align: center">Our Training Signal<div className="imgbox"><img className="center-fit" src="graphics/sr_8_to_16/' + sr_files_8[idx] + '_training.png" width="300"></div>'
            tr3 += '<audio src="' + audio_base_url + 'audio/sr_8_to_16/' + sr_files_8[idx] + '_training.ogg" controls></audio></td>';
            tr3 += '<td style="text-align: center">Extended<div className="imgbox"><img className="center-fit" src="graphics/sr_8_to_16/' + sr_files_8[idx] + '_ours.png" width="300"></div>'
            tr3 += '<audio src="' + audio_base_url + 'audio/sr_8_to_16/' + sr_files_8[idx] + '_ours.ogg" controls></audio></td></tr>';
            $('#sr_table_8_to_16').append(tr);
            $('#sr_table_8_to_16').append(tr2);
            $('#sr_table_8_to_16').append(tr3);
        } else {
            tr = '<tr><th colSpan="4" style="background-color: lightgrey; background-clip: padding-box">' + sr_files_8[idx] + '</th></tr>';
            tr2 = '<tr><td style="text-align: center">Input (Low Resolution)<div className="imgbox"><img className="center-fit" src="graphics/sr_8_to_16/' + sr_files_8[idx] + '_lr.png" width="300"></div>';
            tr2 += '<audio src="' + audio_base_url + 'audio/sr_8_to_16/' + sr_files_8[idx] + '_lr.ogg" controls></audio></td></tr>';
            tr2 += '<tr><td style="text-align: center">GT (High-Resolution)<div className="imgbox"><img className="center-fit" src="graphics/sr_8_to_16/' + sr_files_8[idx] + '_gt.png" width="300"></div>';
            tr2 += '<audio src="' + audio_base_url + 'audio/sr_8_to_16/' + sr_files_8[idx] + '_gt.ogg" controls></audio></td></tr></tr>';
            tr3 = '<tr><td style="text-align: center">Our Training Signal<div className="imgbox"><img className="center-fit" src="graphics/sr_8_to_16/' + sr_files_8[idx] + '_training.png" width="300"></div>';
            tr3 += '<audio src="' + audio_base_url + 'audio/sr_8_to_16/' + sr_files_8[idx] + '_training.ogg" controls></audio></td></tr>';
            tr3 += '<tr><td style="text-align: center">Extended<div className="imgbox"><img className="center-fit" src="graphics/sr_8_to_16/' + sr_files_8[idx] + '_ours.png" width="300"></div>';
            tr3 += '<audio src="' + audio_base_url + 'audio/sr_8_to_16/' + sr_files_8[idx] + '_ours.ogg" controls></audio></td></tr>';
            $('#sr_table_8_to_16_small').append(tr);
            $('#sr_table_8_to_16_small').append(tr2);
            $('#sr_table_8_to_16_small').append(tr3);
        }
    }
    number_of_sr_8_files += j;
    if (number_of_sr_8_files >= sr_files_8.length) {
        $("#add_sr_button_8_to_16_button").hide()
    }
}

function load_inpainting_files() {
    var is_small_screen = $('#inpainting_table').parent().css('display') == 'none';
    var tr = '';
    for (var j = 0; j < 5; j++) {
        var idx = j + number_of_inpainting_files;
        if (idx == inpainting_files.length) {
            $("#add_inpainting_button").hide()
            break
        }
        if (!is_small_screen) {
            tr += '<tr>';
            tr += '<td><audio class="short-audio" src="' + audio_base_url + 'audio/inpainting/' + inpainting_files[idx] + '-with_hole.ogg" controls></audio></td>';
            tr += '<td><audio class="short-audio" src="' + audio_base_url + 'audio/inpainting/' + inpainting_files[idx] + '-real.ogg" controls></audio></td>';
            tr += '<td><audio class="short-audio" src="' + audio_base_url + 'audio/inpainting/' + inpainting_files[idx] + '-gacela.ogg" controls></audio></td>';
            tr += '<td><audio class="short-audio" src="' + audio_base_url + 'audio/inpainting/' + inpainting_files[idx] + '-ours.ogg" controls></audio></td>';
            tr += '</tr>';
            $('#inpainting_table').append(tr);
        } else {
            if (idx % 2 == 1) {
                bg_color = '#f1f1f1';
            } else {
                bg_color = 'white';
            }
            tr = '<tr><td>';
            tr += '<table><tr><th style="background-color: ' + bg_color + '">Input</th><td style="background-color: ' + bg_color + '"><audio class="short-audio" src="' + audio_base_url + 'audio/inpainting/' + inpainting_files[idx] + '-with_hole.ogg" controls></audio></td></tr>';
            tr += '<tr><th style="background-color: ' + bg_color + '">GT</th><td style="background-color: ' + bg_color + '"><audio class="short-audio" src="' + audio_base_url + 'audio/inpainting/' + inpainting_files[idx] + '-real.ogg" controls></audio></td></tr>';
            tr += '<tr><th style="background-color: ' + bg_color + '">GACELA</th><td style="background-color: ' + bg_color + '"><audio class="short-audio" src="' + audio_base_url + 'audio/inpainting/' + inpainting_files[idx] + '-gacela.ogg" controls></audio></td></tr>';
            tr += '<tr><th style="background-color: ' + bg_color + '">Ours</th><td style="background-color: ' + bg_color + '"><audio class="short-audio" src="' + audio_base_url + 'audio/inpainting/' + inpainting_files[idx] + '-ours.ogg" controls></audio></td></tr>';
            tr += '</table></td></tr>';
            $('#inpainting_table_small').append(tr);
        }
    }
    number_of_inpainting_files += j;

    if (number_of_inpainting_files >= inpainting_files.length) {
        $("#add_inpainting_button").hide()
    }
}

function load_denoising_files() {
    var real_audios = [];
    var rec_audios = [];
    var play_buttons = [];
    var switch_buttons = [];
    var labels = [];
    var is_playing = [];
    var is_small_screen = $('#denoising_table').parent().css('display') == 'none';
    for (var j = 0; j < denoising_files.length; j++) {
        if (!is_small_screen) {
            txt = '<tr>';
            txt += '<th style="text-align: center" colspan="3"><a href="' + denoising_urls[j] + '" target="_blank">' + denoising_files[j] + '</a>';
            txt += '<audio src="' + audio_base_url + 'audio/denoising/' + denoising_files[j].replaceAll(' ', '_') + '_real.ogg" id="denoising_real' + String(j) + '"></audio>';
            txt += '<audio muted src="' + audio_base_url + 'audio/denoising/' + denoising_files[j].replaceAll(' ', '_') + '_rec.ogg" id="denoising_rec' + String(j) + '"></audio>';
            txt += '</th>';
            txt += '</tr>';
            txt += '<tr>';
            txt += '<td><div class="imgbox"><img class="center-fit" src="graphics/denoising/' + denoising_files[j].replaceAll(' ', '_') + '_real.png" height="200"></div></td>';
            txt += '<td><div class="imgbox"><img class="center-fit" src="graphics/denoising/' + denoising_files[j].replaceAll(' ', '_') + '_rec.png" height="200"></div></td>';
            txt += '<td><div id="denoising_label' + String(j) + '">Noisy</div><br><button id="denoising_play' + String(j) + '">Play/Pause</button><button id="denoising_switch' + String(j) + '">Switch</button></td>';
            txt += '</tr>';
            $('#denoising_table').append(txt);
        } else {
            txt = '<tr>';
            txt += '<th style="text-align: center; background-color: #f1f1f1" colspan="3"><a href="' + denoising_urls[j] + '" target="_blank">' + denoising_files[j] + '</a>';
            txt += '<audio src="' + audio_base_url + 'audio/denoising/' + denoising_files[j].replaceAll(' ', '_') + '_real.ogg" id="denoising_real' + String(j) + '"></audio>';
            txt += '<audio muted src="' + audio_base_url + 'audio/denoising/' + denoising_files[j].replaceAll(' ', '_') + '_rec.ogg" id="denoising_rec' + String(j) + '"></audio>';
            txt += '</th></tr>';
            txt += '<tr><td style="background-color: white"><div class="imgbox"><img class="center-fit" src="graphics/denoising/' + denoising_files[j].replaceAll(' ', '_') + '_real.png" width="400"></div>';
            txt += '<div class="imgbox"><img class="center-fit" src="graphics/denoising/' + denoising_files[j].replaceAll(' ', '_') + '_rec.png" width="400"></div>';
            txt += '<div id="denoising_label' + String(j) + '">Noisy</div><br><button id="denoising_play' + String(j) + '">Play/Pause</button><button id="denoising_switch' + String(j) + '">Switch</button>';
            txt += '</td></tr>';
            $('#denoising_table_small').append(txt);
        }
        real_audios.push($("#denoising_real" + String(j)));
        rec_audios.push($("#denoising_rec" + String(j)));
        play_buttons.push($("#denoising_play" + String(j)));
        switch_buttons.push($("#denoising_switch" + String(j)));
        is_playing.push(false);
        labels.push($("#denoising_label" + String(j)));
        play_buttons[j].click(function () {
            var idx = parseInt(this.id[this.id.length - 1]);
            if (!is_playing[idx]) {
                real_audios[idx][0].play();
                rec_audios[idx][0].play();
                is_playing[idx] = true
            } else {
                real_audios[idx][0].pause();
                rec_audios[idx][0].pause();
                is_playing[idx] = false
            }
        });
        switch_buttons[j].click(function () {
            var idx = parseInt(this.id[this.id.length - 1]);
            if (real_audios[idx][0].muted) {
                real_audios[idx].prop("muted", false);
                rec_audios[idx].prop("muted", true);
                labels[idx].text('Noisy')
            } else {
                real_audios[idx].prop("muted", true);
                rec_audios[idx].prop("muted", false);
                labels[idx].text('Denoised')
            }
        })
    }

}

function load_limitations_files() {
    var is_small_screen = $('#limitations_table').parent().css('display') == 'none';
    var explanations = ['High pitched speech -> degraded quality', 'High pitched speech -> degraded quality', 'High pitched musical instruments -> degraded quality', 'Noisy reverberant speech -> high pitched noise', 'Reverberant speech -> high pitched noise']
    for (var j = 0; j < 5; j++) {
        var idx = j + number_of_limitations_files;
        if (idx == limitations_files.length) {
            break
        }
        if (!is_small_screen) {
            tr = '<tr>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/limitations/' + limitations_files[idx] + '_real.ogg" controls></audio>' +
                '</td>';
            tr += '<td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/limitations/' + limitations_files[idx] + '_fake.ogg" controls></audio>' +
                '</td>';
            tr += '<td>' + explanations[idx] + '</td>';
            tr += '</tr>';
            $('#limitations_table').append(tr);
        } else {
            tr = '<tr>';
            tr += '<th>Real (Input)</th>';
            tr += '<tr><td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/limitations/' + limitations_files[idx] + '_real.ogg" controls></audio>' +
                '</td></tr>';
            tr += '<th>Fake</th>';
            tr += '<tr><td style="text-align: center">' +
                '<audio class="short-audio" src="' + audio_base_url + 'audio/limitations/' + limitations_files[idx] + '_fake.ogg" controls></audio><br>' + explanations[idx] +
                '</td></tr>';
            $('#limitations_table_small').append(tr);
        }
    }
    number_of_limitations_files += j;
}

$(document).ready(function () {
    // Inpainting Table
    $.get('inpainting_files.dat', function (data) {
        inpainting_files = data.split(',');
        load_inpainting_files()
        $("#add_inpainting_button").show()
    });
    $("#add_inpainting_button").click(function () {
        load_inpainting_files();
    });

    // Songs table
    $("#add_music_button").hide();
    $.get('music_files.dat', function (data) {
        music_files = data.split('\n')[0].split(',');
        music_names = data.split('\n')[1].split(',');
        music_artists = data.split('\n')[2].split(',');
        music_links = data.split('\n')[3].split(',');
        music_offsets = data.split('\n')[4].split(',');
        load_music_files();
        $("#add_music_button").show()
    });
    $("#add_music_button").click(function () {
        load_music_files()
    });

    // RF table
    create_rf_table()

    // SR table
    $("#add_sr_button").hide();
    $.get('sr_files.dat', function (data) {
        sr_files = data.split(',');
        var highlights = ['p347_410', 'p360_323', 'p362_120', 'p364_153', 'p374_388'];
        for (k = 0; k < highlights.length; k++) {
            var idx = sr_files.indexOf(highlights[k]);
            sr_files.splice(k, 0, sr_files.splice(idx, 1)[0]);
        }
        load_sr_files();
        $("#add_sr_button").show()
    });
    $("#add_sr_button").click(function () {
        load_sr_files();
    });

    // SR table (8KHz -> 16KHz)
    $("#add_sr_button_8_to_16").hide();
    $.get('sr_files_8_to_16.dat', function (data) {
        sr_files_8 = data.split('\n')[0].split(',');
        var highlights = ['p374_388', 'p364_153', 'p360_323', 'p361_255', 'p351_341', 'p362_120'];
        for (k = 0; k < highlights.length; k++) {
            var idx = sr_files_8.indexOf(highlights[k]);
            sr_files_8.splice(k, 0, sr_files_8.splice(idx, 1)[0]);
        }
        load_sr_files_8();
        $("#add_sr_button_8_to_16").show()
    });
    $("#add_sr_button_8_to_16").click(function () {
        load_sr_files_8();
    });

    // Speeches table
    $.get('speeches_files.dat', function (data) {
        speeches_files = data.split(',');
        var highlights = ['FastLife_for_submission_67', 'LoseYourself_for_submission_71', 'trump_farewell_address_8_8'];
        for (k = 0; k < highlights.length; k++) {
            var idx = speeches_files.indexOf(highlights[k]);
            speeches_files.splice(k, 0, speeches_files.splice(idx, 1)[0]);
        }
        load_speeches_files();
        $("#add_speeches_button").show()
    });
    $("#add_speeches_button").click(function () {
        load_speeches_files()
    });

    // Instruments
    $("#add_instruments_button").hide();
    $.get('instruments_files.dat', function (data) {
        instruments_files = data.split('\n')[0].split(',');
        instruments_names = data.split('\n')[1].split(',');
        var highlights = ['185_TenorSaxophone_mdb_for_submission_66', '27_violin_for_submission_54', '25_DistortedElectricGuitar_mdb_for_submission_62', '2_saxophone_for_submission_53']
        for (k = 0; k < highlights.length; k++) {
            var idx = instruments_files.indexOf(highlights[k]);
            instruments_files.splice(k, 0, instruments_files.splice(idx, 1)[0]);
            instruments_names.splice(k, 0, instruments_names.splice(idx, 1)[0]);
        }

        load_instruments_files();
        $("#add_instruments_button").show()
    });
    $("#add_instruments_button").click(function () {
        load_instruments_files()
    });

    // Ambient
    $("#add_ambient_button").hide();
    $.get('ambient_files.dat', function (data) {
        ambient_files = data.split('\n')[0].split(',');
        load_ambient_files()
    });

    // Denoising
    $.get('denoising_files.dat', function (data) {
        denoising_files = data.split('\n')[0].split(',');
        denoising_urls = data.split('\n')[1].split(',');
        load_denoising_files()
    });

    // Limitations
    $.get('limitations_files.dat', function (data) {
        limitations_files = data.split('\n')[0].split(',');
        load_limitations_files()
    });
});