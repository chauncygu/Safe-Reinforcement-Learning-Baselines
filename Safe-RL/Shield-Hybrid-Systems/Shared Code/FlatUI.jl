## Right no sorry. This isn't a UI library or anything. The color theme is just called that.

colors = 
	(TURQUOISE = colorant"#1abc9c", 
	EMERALD = colorant"#2ecc71", 
	PETER_RIVER = colorant"#3498db", 
	AMETHYST = colorant"#9b59b6", 
	WET_ASPHALT = colorant"#34495e",
	
	GREEN_SEA   = colorant"#16a085", 
	NEPHRITIS   = colorant"#27ae60", 
	BELIZE_HOLE  = colorant"#2980b9", 
	WISTERIA     = colorant"#8e44ad", 
	MIDNIGHT_BLUE = colorant"#2c3e50", 
	
	SUNFLOWER = colorant"#f1c40f",
	CARROT   = colorant"#e67e22",
	ALIZARIN = colorant"#e74c3c",
	CLOUDS   = colorant"#ecf0f1",
	CONCRETE = colorant"#95a5a6",
	
	ORANGE = colorant"#f39c12",
	PUMPKIN = colorant"#d35400",
	POMEGRANATE = colorant"#c0392b",
	SILVER = colorant"#bdc3c7",
	ASBESTOS = colorant"#7f8c8d")

trans = [colorant"#FFFFFF", colorant"#5BCEFA", colorant"#F5A9B8"]
enby =  [colorant"#FCF434", colorant"#FFFFFF", colorant"#9C59D1", colorant"#2C2C2C"]

transitioncolors = trans
transitionlabels = ["Initial", "Reachable", "Not reachable"]

bbshieldlabels = 	["{hit, nohit}", "{hit}", "{}"]
bbshieldcolors = 	[enby[2], enby[3], enby[4]]

rwshieldlabels = 	["{fast, slow}", "{fast}", "{}"]
rwshieldcolors = 	[enby[2], enby[3], enby[4]]

opshieldlabels = ["{}", "{off}", "{on}", "{on, off}"]
opshieldcolors = [enby[4], enby[3], enby[1], enby[2]]

dcshieldlabels = ["{}", "{off}", "{on}", "{on, off}"]
dcshieldcolors = [enby[4], enby[3], enby[1], enby[2]]

ccshieldcolors=[
	enby[4],			# 000
	enby[3],			# 001
	colors.EMERALD, 	# 010 
	enby[1],			# 011
	colors.TURQUOISE, 	# 100
	colors.CARROT, 		# 101
	colors.PETER_RIVER, # 110
	enby[2],			# 111
]

ccshieldlabels=["{}", "{forwards}", "{neutral}", "{neutral, forwards}", "{backwards}", "{backwards, forwards}", "{backwards, neutral}", "{backwards, neutral, forwards}"]

ccshieldlabelsabbreviated = ["{}", "{backwards}", "{backwards, neutral}", "{backwards, neutral, forwards}"]

# Used for shielding results figures.
shielding_type_colors = (pre_shielded=colors.GREEN_SEA, no_shield=colors.BELIZE_HOLE, post_shielded=colors.SUNFLOWER, layabout=colors.MIDNIGHT_BLUE)
interventions_colors = [colors.TURQUOISE colors.PETER_RIVER colors.EMERALD colors.AMETHYST]		# d=0, d=10, d=100, d=1000
deaths_colors = [colors.PETER_RIVER colors.EMERALD colors.AMETHYST]		# d=10, d=100, d=1000