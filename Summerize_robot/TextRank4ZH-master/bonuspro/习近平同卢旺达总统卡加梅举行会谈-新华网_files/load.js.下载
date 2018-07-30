
function kkLoadPlayer(playerDiv,sublistDiv,errorDiv){
	if(!/AppleWebKit.*Mobile|iPad|iPhone|iPod|Android/i.test(navigator.userAgent)){
		document.write('<script type="text/javascript" src="http://misc.vod.news.cn/v/misc/swfobject.js" charset="gbk"></script>');
		document.write('<script type="text/javascript" src="http://misc.vod.news.cn/v/misc/player_min.js?v=1.1" charset="gbk"></script>');
		document.write('<script type="text/javascript" src="http://misc.vod.news.cn/v/misc/core_min.js?v=1.1" charset="gbk"></script>');
		if(typeof playerDiv!='undefined'&&playerDiv!=null){
			document.getElementById(playerDiv).innerHTML = '<div id="_player" style="height: 100%; width: 100%;"></div>';
		}else{
			document.write('<div class="player"><div id="_player" class="playerbox playerbox_wide"></div>');
		}
		if(typeof sublistDiv!='undefined'&&sublistDiv!=null){
			document.getElementById(sublistDiv).innerHTML = '<dl id="dl_subList" class="w_zy"><dt style="display:;" id="dt_fenji">ּ</dt><dd><ul id="ul_subLabel"></ul></dd></dl><div class="zy_list" id="div_progPanel"><ul id="ul_progList"></ul></div>';
		}
		if(typeof errorDiv!='undefined'&&errorDiv!=null){
			document.getElementById(errorDiv).innerHTML = '<div class="errorbox" style="display:none;" id="errorDiv"></div>';
		}else if(playerDiv){
			var cont = document.getElementById(playerDiv).parentNode,
				tmp = document.createElement("div");
			tmp.style.display = "none";
			tmp.id = "errorDiv";
			tmp.className = "errorbox";
			cont.insertBefore(tmp, cont.firstChild);
		}
		var mode = 0;


		if(/mac|opera/i.test(navigator.userAgent)){mode=2}
		kkInit(mode);
		
	}else{
		document.write('<script type="text/javascript" src="http://misc.vod.news.cn/v/misc/mplayer_min.js?v=1.1" charset="gbk"></script>');
		document.write('<script type="text/javascript" src="http://misc.vod.news.cn/v/misc/core_min.js?v=1.1" charset="gbk"></script>');
		if(typeof playerDiv!='undefined'&&playerDiv!=null){
			document.getElementById(playerDiv).innerHTML = '<div id="_player" style="height: 100%; width: 100%;"></div>';
		}else{
			document.write('<div class="player"><div id="_player" class="playerbox playerbox_wide"></div>');
		}
		if(typeof errorDiv!='undefined'&&errorDiv!=null){
			document.getElementById(errorDiv).innerHTML = '<div class="errorbox" style="display:none;" id="errorDiv"></div>';
		}else if(playerDiv){
			var cont = document.getElementById(playerDiv).parentNode,
				tmp = document.createElement("div");
			tmp.style.display = "none";
			tmp.id = "errorDiv";
			tmp.className = "errorbox";
			cont.insertBefore(tmp, cont.firstChild);
		}
		var mode = 5;
		kkInit(mode);
	}
	

}

function kkInit(playMode){
	if(typeof kkCore=='undefined'||!kkCore||!document.getElementById('_player') || !window.G_PLAYER_INIT || typeof G_PLUGIN_MOVIE=='undefined'){
		setTimeout(function(){kkInit(playMode);},100);
		return false;
	}else{
		document.getElementById('_player').style.top='-2000px';
		G_PLAYER_INIT.printObject("_player",false,'100%','100%',null,'&windowMode=webapp&ext=1');
		G_CORE_INIT = new kkCore.Init(playMode);
		checkPlayerLoaded();
	}
}

function checkPlayerLoaded (){
	if(G_PLAYER_INIT.playerLoaded){
		kkPlayerLoaded();
		return true;
	}else{
		setTimeout(function(){checkPlayerLoaded();},100);
	}
}
