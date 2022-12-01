# data-to-text-baseline
Implementation of Text-to-Text Pre-Training for Data-to-Text Tasks on MultiWOZ dataset

## Overview
This is an implementation of the paper, [Text-to-Text Pre-Training for Data-to-Text Tasks](https://arxiv.org/pdf/2005.10433v3.pdf), only on MultiWOZ dataset.

The paper suggests a baseline for Data-to-Text task by fine-tuning T5-base model. 

![image](imgs/t5_model.png)

It is simple yet achieves SOTA on MultiWOZ dataset.
Pairs of _act_ and _system utterence_ from MultiWOZ dataset is each inputs and outputs of the model.
Especially, _act_ is linearized from its structure like the following image.

<!-- image -->
![image](imgs/linearized_data.png)

Then, the linearized act is prefixed with prompt such as `data to text:`.

## Install

```bash
git clone --recurse-submodules https://github.com/yjang43/data-to-text-baseline.git
cd data-to-text-baseline
pip install -r requirements.txt
```


## Result
### BLEU Score
![image](imgs/bleu_score.png)
_BLEU score was lower than what is reported in the paper.
Note that thorough hyperparameter is not done in this project._

### Loss Graph
![image](imgs/loss_graph.png)


### Generated Samples
20 samples from dev set after 5k steps with 128 batch size.

|act|pred_utt                     |gt_utt|
|---|-----------------------------|------|
|data to text: Hotel Request Price = ? &#124; general greet none = none|I can help with that. Do you have a price range in mind?|Okay, I'll help you with that. Do you have a price range in mind?|
|data to text: Hotel Inform Parking = none , Name = The Acorn Guest House , Internet = none &#124; general reqmore none = none|The Acorn Guest House has free parking and wifi. Would you like more information?|The Acorn Guest House offers free parking as well as free Wifi. Can I help with anything else? |
|data to text: general reqmore none = none &#124; Booking Book Ref = 1KAYW650|Booking was successful. The table will be reserved for 15 minutes. Reference number is : 1| Booking was successful. Reference number is : 1KAYW650. Can I help you with anything else?|
|data to text: Booking Request People = ? , Day = ?|How many people will be dining and what day will you be arriving?|How many people will be staying and what is the starting day you will need?|
|data to text: general reqmore none = none &#124; Booking Book Ref = AK8Y5GX6|Booking was successful. The table will be reserved for 15 minutes. Reference number is :|Booking was successful. The table will be reserved for 15 minutes. Reference number is : AK8Y5GX6. Can I help with anything else today?|
|data to text: Booking Inform none = none &#124; Hotel Inform Price = moderate , Name = The Acorn Guesthouse , Parking = none , Area = north|The Acorn Guesthouse is in the north part of town and has a moderate price range|The Acorn Guesthouse located in the north offers free parking and is in moderate pricing range. Would you like me to book a room for you?|
|data to text: general bye none = none &#124; general welcome none = none|You're welcome. Have a great day!|Thank you for contacting us and have a nice day.|
|data to text: Booking Request Stay = ?|How many nights will you be staying?|of course, how many nights will you be staying?|
|data to text: Attraction Inform Fee = free , Addr = lensfield road|The address is lensfield road and the entrance fee is free.|The address is lensfield road and the admission is free.|
|data to text: Train Inform Leave = 9:29 , Arrive = 10:07 &#124; Train OfferBook none = none|There is a train leaving at 9:29 and arriving at 10:07. Would you like|There is a train leaving at 9:29 and arriving by 10:07. Would you like me to book that for you?|
|data to text: Train Inform Choice = 10 &#124; Train Request Arrive = ? , Leave = ?|There are 10 trains that meet your criteria. Do you have a preferred departure or arrival time|I have 10 trains that match your criteria. What time would you like to leave or arrive by?|
|data to text: Train Request Leave = ? , Dest = ?|What is your destination and what time would you like to leave?|And can you tell me a preferred time to travel and where you will travelling to?|
|data to text: Train OfferBooked Ref = WI8UR8MB , Ticket = 10.1 GBP|Booking was successful, the total fee is 10.1 GBP payable at the station. Reference|Booking was successful, the total fee is 10.1 GBP payable at the station . Reference number is : WI8UR8MB.|
|data to text: general reqmore none = none &#124; Attraction Inform Type = museum|It is a museum. Is there anything else I can help you with?|It is a museum. Would you like more information on it?|
|data to text: general welcome none = none|You're welcome, have a great day!|You're welcome, glad I could help.|
|data to text: Train Request Day = ?|What day would you like to travel?|i sure can! what day are you traveling?|
|data to text: Hotel Inform Area = north , Choice = two &#124; Booking Request People = ?|There are two in the north. How many people will be staying?|I have two options that are in the north area. How many people would you like to reserve a room for?|
|data to text: Train Inform Time = 79 minutes &#124; general reqmore none = none|The travel time is 79 minutes. Is there anything else I can help you with?|Absolutely. The total travel time is 79 minutes. Is there anything else I can help you with?|
|data to text: Train Inform Arrive = 15:55 , Dest = Norwich , Depart = Cambridge , Leave = 14:36 &#124; Train OfferBook none = none|There is a train leaving Cambridge at 14:36 and arriving in Norwich at 15:55|Train TR5190 leaves Cambridge at 14:36 and arrives in Norwich at 15:55. Would you like me to book it for you?|
|data to text: Train Request Day = ? , Leave = ?|What day and time would you like to travel?|What is your departure day and time?|
