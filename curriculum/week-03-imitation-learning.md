# Imitation Learning


prerequire

notations of imitation learning or intellegent systems or learning systems in general

at - action 
ot - observations
st - stae

t - trajectory

r(s,a) - reward


example ??


imitation learning: given set of traejcotries collected by an "expert" called as "demonstration"

D = {(s1,a1,....,st)}



the goal is to learn a policy Pi that imitates the expert behaviour



i.e beheaviour cloning

given D={(s1,a1....st)}

for dtereministic policy regress to expert actions

mintheta 1/|d| sigma(s,a)bnelongsD||a-a||^2 where a=pi0(s)

deploy policy p0 on the robot


does it work?

cite end ot end learning for selfd dsirivng cars bojarski et all 2016



add fake data that illustrates correction with side facing cameras 


"data augmenetation"



the upper bound of beuhaviour cloning 

add the figurte of actiona nbd state drift
for time horizon T


districvtubion shift causes the error to grow quadratically

"a reduction of iomiation learning and structured prediction to no regret online learning" by ross el all 2011


addressing compounding error

how can we make p(expert) = ppi(s)

states visited expert  = staes visited by the policy

 
flow of how dagger works 

rollout pi0 -> quer label expert action at visited states a* -> aggregate coorectio with exeisting data D<-DU{(s`,a*)}
update policy <- arg min L{pi0, D}







goods
lets the human take the control, the true expert
paradox bc works better if the data has more mistakes nad recoeries
hence algho with converge pi(s) = p expert
achives O(t) instead of BC 0(<AT^2)


bads
how to detect when anintervention is needed
hindsight laberlling and expertt query difficuilt this is not ideal





exmaple waymo self driving


why mght we still fail to mimick the exeprt

action depends only on current observation, human bejhaviour mightbe afffected by past observertions, emotions aand privileged informatione tc




also human obs != robot caerma observation


