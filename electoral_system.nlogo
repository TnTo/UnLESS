; work environment
; NetLogo 6.0.4
; Python 3.7
; py extension 0.4.1 modified as stated at https://github.com/NetLogo/Python-Extension/issues/14
; NumPy 1.16.3

; import useful extensions
; py is a NetLogo -> Python bridge
extensions [py]

; define two types of agents
breed [electors elector]
breed [parties party]

; define additional variables for agents
patches-own [n_electors_on votes_parties_on]
parties-own [votes previous_votes tmp_votes seats previous_seats age time_without_seats next_move old_pos pos]

to setup
  setup_environment
  setup_electors
  setup_parties
  setup_io

  ; restart clock
  reset-ticks
end

to setup_environment
  clear-all

  ; setup py extension
  py:setup py:python
  py:run "import numpy as np"
  py:run "import neural_network as nn"

  ; inizialize random seed
  random-seed seed
  py:set "seed" seed
  py:run "np.random.seed(seed)"

  ; load NN
  load_nn
end

to load_nn
  py:set "nn_path" nn_path
  py:run "nnet = nn.load_nn({'nn_path':str(nn_path)})"
end

to setup_electors
  ; initialize electors
  set-default-shape electors "dot"
  create-electors n_electors

  ; beta distributed random number are generated with numpy
  ; set beta distribution parameters
  ifelse x_opinion = "bimodal"
  [ py:set "a" 0.5
    py:set "b" 0.5 ]
  [ ifelse x_opinion = "unimodal"
    [ py:set "a" 3
      py:set "b" 3 ]
    ; uniform case managed as default
    [ py:set "a" 1
      py:set "b" 1 ]]
  ifelse y_opinion = "bimodal"
  [ py:set "c" 0.5
    py:set "d" 0.5 ]
  [ ifelse y_opinion = "unimodal"
    [ py:set "c" 3
      py:set "d" 3 ]
    ; uniform case managed as default
    [ py:set "c" 1
      py:set "d" 1 ]]

  ask electors [
    ; hide the electors?
    set hidden? (not show_electors)
    set color yellow
    ; set electors position - note that our space is [-10,10] instead of [0,1]
    setxy (py:runresult "20 * np.random.beta(a,b) - 10") (py:runresult "20 * np.random.beta(c,d) - 10")
  ]

  ; assign to each patches the number of electors on it
  ask patches [
    set n_electors_on (count electors-here)
  ]
  ; color patches if electors are not showed
  if (not show_electors) [
    let max_electors_on (max [n_electors_on] of patches)
    ask patches [
      set pcolor scale-color yellow n_electors_on 0 max_electors_on
  ]]
end

to setup_parties
  ask parties [
    die
  ]

  ; inizialize parties
  create-parties n_parties_t0 [
    setxy ((random-float 20) - 10) ((random-float 20) - 10)
    set color red
    set size 0.5
    set heading 0
  ]
end

to setup_io
  ; open file to save data
  file-close
  let file_path ""
  ifelse data_path = "" [
    set file_path (word "data/" (date-and-time) ".csv")
  ] [
    set file_path (data_path)
  ]
  ifelse append [
    file-open file_path
  ] [
    carefully [
      file-delete file_path
      file-open file_path
    ] [
      file-open file_path
    ]
  ]
  file-open file_path

end

to go
  ; eventually add a new party
  add_party

  ; allow parties to move
  move

  ; ask electors to vote
  vote

  ; distribute seats among the parties and eventually remove some of them
  distribute_seats

  save_data

  ; manage the clock
  if ticks = max_ticks [
    file-close
    stop
  ]
  tick
end

to add_party
  if ((random-float 1) < p_new_party) [
    create-parties 1 [
      setxy ((random-float 20) - 10) ((random-float 20) - 10)
      set color red
    ]
  ]
end

to move
  ; save the input for the NN about the state of the world which is equal for each party

  ifelse parties_see_electors [
  ; if parties see electors save the number of electors on each patch
    py:set "state" (map [x -> [n_electors_on] of x] (sort patches))
  ] [
  ; if not save the votes received from the partties on each patch (i.e. the result of last election and the position of the parties)
    py:set "state" (map [x -> [votes_parties_on] of x] (sort patches))
  ]

  ask parties [

    set old_pos list xcor ycor

    ; parties has a small probability to move randomly to explore new possibility (so called epsilon-greedy behaviour)
    ; epsilon = 1 is always manage by the NN, epsilon = 0 is always random
    ifelse ((random-float 1) < (epsilon)) [
      ; complete the input with the party position
      py:set "my_pos" (list xcor ycor)
      set next_move (py:runresult "nn.next_move(nnet, state, my_pos)")
    ][
      set next_move (random 5)
    ]
    ; moves dictionary
    ; 0 no move
    ; 1 forward
    ; 2 left
    ; 3 back
    ; 4 right
    if (not (next_move = 0)) [
      left (90 * (next_move - 1)) ; turn
      forward step_distance       ; move
    ]

    set heading 0

    set pos list xcor ycor
  ]
end

to vote
  ; save votes at previous election because we need them for learning algorithm
  ask parties [
    set previous_votes votes
    set votes 0
  ]

  ; ask electors to vote
  ask electors [
    let nearest_party min-one-of parties [max list abs (xcor - [xcor] of myself) abs (ycor - [ycor] of myself) ]
    let distance_from_nearest_party (max list abs ([xcor] of nearest_party - xcor) abs ([ycor] of nearest_party - ycor))
    ; electors are more likely to vote a party more it is near
    ; electors don't vote for parties farer than 10 (i.e. half of the space)
    if (distance_from_nearest_party < 10) and (random-float 10 > distance_from_nearest_party) [
      ask nearest_party [
        set votes (votes + 1)
      ]
    ]
  ]

  ask patches [
    set votes_parties_on sum [votes] of parties-here
  ]
end

to distribute_seats
  ask parties [
    set previous_seats seats
    set seats 0
    set tmp_votes votes
  ]

  ; distribute seats
  if electoral_system = "d'Hondt" [
    while [sum [seats] of parties < total_seats] [
      ask max-one-of (parties with-max [tmp_votes]) [votes] [
        set seats (seats + 1)
        set tmp_votes (tmp_votes / (seats + 1))
      ]
    ]
  ]
  if electoral_system = "plurality" [
    ask max-one-of parties [votes] [
      set seats total_seats
    ]
  ]

  ; update size and color and remove parties if needed
  ask parties [
    set age (age + 1)
    ifelse seats = 0 [
      set time_without_seats (time_without_seats + 1)
      if (age > safe_time_after_birth) and (time_without_seats > time_without_seats_before_death) [
        die
      ]
      set color red
      set size 0.5
    ] [
      set color green
      set size (((seats / total_seats) * 1.5) + 0.5)
    ]
  ]
end

to save_data
  let electors_state (map [x -> [n_electors_on] of x] (sort patches))
  let votes_state (map [x -> [votes_parties_on] of x] (sort patches))
  ask parties [
    file-print (word ticks ";" who ";" old_pos ";" pos ";" electors_state ";" votes_state ";" next_move ";" previous_votes ";" votes ";" previous_seats ";" seats)
  ]
end

; This is necessary to be used with pyNetLogo
; The go button already loops the go function
to loop_go
  loop [
    go
    if (ticks = max_ticks) [
      stop
    ]
  ]
end
@#$#@#$#@
GRAPHICS-WINDOW
21
290
449
719
-1
-1
20.0
1
10
1
1
1
0
0
0
1
-10
10
-10
10
0
0
1
ticks
30.0

SLIDER
19
24
191
57
n_electors
n_electors
0
2000
1000.0
1
1
NIL
HORIZONTAL

SLIDER
203
24
448
57
n_parties_t0
n_parties_t0
0
20
1.0
1
1
NIL
HORIZONTAL

SLIDER
203
66
449
99
p_new_party
p_new_party
0
1
0.0
0.025
1
NIL
HORIZONTAL

CHOOSER
21
72
159
117
x_opinion
x_opinion
"uniform" "unimodal" "bimodal"
1

CHOOSER
21
132
159
177
y_opinion
y_opinion
"uniform" "unimodal" "bimodal"
0

SLIDER
204
107
450
140
safe_time_after_birth
safe_time_after_birth
0
10
1.0
1
1
NIL
HORIZONTAL

SLIDER
203
146
450
179
time_without_seats_before_death
time_without_seats_before_death
1
500
500.0
1
1
NIL
HORIZONTAL

CHOOSER
470
25
616
70
electoral_system
electoral_system
"d'Hondt" "plurality"
1

SLIDER
470
76
617
109
total_seats
total_seats
1
100
1.0
1
1
NIL
HORIZONTAL

SLIDER
204
190
452
223
step_distance
step_distance
0
10
0.5
0.1
1
NIL
HORIZONTAL

SWITCH
205
234
450
267
parties_see_electors
parties_see_electors
0
1
-1000

BUTTON
821
24
882
57
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
821
64
882
97
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

INPUTBOX
632
25
806
85
seed
1234.0
1
0
Number

SLIDER
632
95
804
128
max_ticks
max_ticks
0
500
500.0
1
1
NIL
HORIZONTAL

SWITCH
20
193
160
226
show_electors
show_electors
1
1
-1000

SLIDER
630
143
802
176
epsilon
epsilon
0
1
0.95
0.05
1
NIL
HORIZONTAL

PLOT
494
327
807
447
n_parties
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 2 -16777216 true "" "plot count parties"

PLOT
493
453
807
573
voters (% of electors)
NIL
NIL
0.0
10.0
0.0
1.0
true
false
"" ""
PENS
"default" 1.0 2 -16777216 true "" "plot (sum [votes] of parties) / n_electors"

PLOT
491
580
807
700
average seats per party
NIL
NIL
0.0
10.0
0.0
5.0
true
false
"" ""
PENS
"default" 1.0 2 -16777216 true "" "plot mean [seats] of parties"

INPUTBOX
631
189
805
249
nn_path
nn.h5
1
0
String

INPUTBOX
630
255
806
315
data_path
data/test.csv
1
0
String

SWITCH
510
275
622
308
append
append
0
1
-1000

@#$#@#$#@
## WHAT IS IT?

This model represent some parties which explore different political position to maximize the votes received.
The space could be interpretated using the axis "Democracy-Dictatorship" and "Free Market-Planned Economy", as in Political Compass project.
The aim of the model is to find how electoral system and electors distribution change the optimal party's strategy.

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.0.4
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
