set -x

cores_list=(32 48)
#cores_list=(48)
batch_size_list=(1)
#input_list=(10 32 1024 2012)
input_list=(100)
output_list=(32, 100)
beam_list=(1)



function main() {
    conda_env="$1"
    model="$2"
    working_dir="$3"
    log_prefix="$4"
    compiler_version="$5"
    export log_prefix=${log_prefix}
    validation_script="${WORKSPACE}/lpot-validation/nlp-toolkit/scripts/run_cpp_graph.py"
    CONFIG_PATH="${WORKSPACE}/lpot-validation/nlp-toolkit/scripts/prompt.json"
    # init params
    if [[ "${model}" == "llama-7b-hf" ]]; then
        convert_script="${working_dir}/scripts/convert_llama.py"
        quant_script="./build/bin/quant_llama"
        infer_cmd="./build/bin/run_llama"
        input_model="/tf_dataset2/models/nlp_toolkit/llama-7b-hf"
        precision_list=("q4_j_vnni_b128" "q4_j_vnni_bf16_b32" "q4_j_vnni_b32" "q4_0" "q4_j_vnni_b128_asym")
        echo "is llama"
    elif [[ "${model}" == "llama-2-7b-chat" ]]; then
        convert_script="${working_dir}/scripts/convert_llama.py"
        quant_script="./build/bin/quant_llama"
        infer_cmd="./build/bin/run_llama"
        input_model="/tf_dataset2/models/nlp_toolkit/llama-2-7b-chat"
        precision_list=("q4_j_vnni_b128" "q4_j_vnni_bf16_b32" "q4_j_vnni_b32" "q4_0")
    elif [[ "${model}" == "gpt-neox-20b" ]]; then
        convert_script="${working_dir}/scripts/convert_gptneox.py"
        quant_script="./build/bin/quant_gptneox"
        infer_cmd="./build/bin/run_gptneox"
        input_model="/mnt/disk1/data2/zhenweil/models/gpt_neox/gpt-neox-20b/"
        precision_list=("q4_j_b128" "q4_j_channel" "q8_j_b128" "q8_j_channel")
    elif [[ "${model}" == "mpt-7b" ]]; then
        convert_script="${working_dir}/scripts/convert_mpt.py"
        quant_script="./build/bin/quant_mpt"
        infer_cmd="./build/bin/run_mpt"
        input_model="/tf_dataset2/models/nlp_toolkit/mpt-7b"
        precision_list=("q4_j_b128" "q4_j_b32" "q4_0")
    elif [[ "${model}" == "falcon-7b" ]]; then
        convert_script="${working_dir}/scripts/convert_falcon.py"
        quant_script="./build/bin/quant_falcon"
        infer_cmd="./build/bin/run_falcon"
        input_model="/tf_dataset2/models/nlp_toolkit/falcon-7b"
        precision_list=("q4_j_b128" "q4_j_b32" "q4_0")
    elif [[ "${model}" == "gptj-6b" ]]; then
        convert_script="${working_dir}/scripts/convert_gptj.py"
        quant_script="./build/bin/quant_gptj"
        infer_cmd="./build/bin/run_gptj"
        model_name="EleutherAI/gpt-j-6b"
        input_model="/tf_dataset2/models/pytorch/gpt-j-6B"
        precision_list=("q4_j_b128" "q4_j_b32" "q4_0" "q4_j_b128_asym")
    elif [[ "${model}" == "starcoder-3b" ]]; then
        convert_script="${working_dir}/scripts/convert_starcoder.py"
        quant_script="./build/bin/quant_starcoder"
        infer_cmd="./build/bin/run_starcoder"
        model_name="bigcode/starcoder"
        input_model="/tf_dataset2/models/pytorch/starcode_3b"
        precision_list=("q4_j_b128" "q4_j_b32" "q4_0")
    elif [[ "${model}" == "bloom-7b" ]]; then
        convert_script="${working_dir}/scripts/convert_bloom.py"
        quant_script="./build/bin/quant_bloom"
        infer_cmd="./build/bin/run_bloom"
        model_name="bigscience/bloom-7b1"
        input_model="/tf_dataset2/models/pytorch/bloom-7b1"
        precision_list=("q4_j_b128" "q4_j_b32" "q4_0")
    elif [[ "${model}" == "opt-1.3b" ]]; then
        convert_script="${working_dir}/scripts/convert_opt.py"
        quant_script="./build/bin/quant_opt"
        infer_cmd="./build/bin/run_opt"
        model_name="facebook/opt-1.3b"
        input_model="/tf_dataset2/models/pytorch/opt-1.3b"
        precision_list=("q4_j_b128" "q4_j_b32" "q4_0")
    elif [[ "${model}" == "dolly-v2-3b" ]]; then
        convert_script="${working_dir}/scripts/convert_dolly.py"
        quant_script="./build/bin/quant_dolly"
        infer_cmd="./build/bin/run_dolly"
        model_name="databricks/dolly-v2-3b"
        input_model="/mnt/disk1/data2/linjun/model/dolly/dolly-v2-3b/"
        precision_list=("q4_j_b128" "q4_j_b32" "q4_0")
    elif [[ "${model}" == "chatglm2" ]]; then
        convert_script="${working_dir}/scripts/convert_chatglm.py"
        quant_script="./build/bin/quant_chatglm2"
        infer_cmd="./build/bin/run_chatglm2"
        model_name="THUDM/chatglm2-6b"
        input_model="/tf_dataset2/models/pytorch/chatglm2-6b"
        precision_list=("q4_j_b128" "q4_j_b32" "q4_0")
    elif [[ "${model}" == "chatglm-6b" ]]; then
        convert_script="${working_dir}/scripts/convert_chatglm.py"
        quant_script="./build/bin/quant_chatglm"
        infer_cmd="python ./scripts/run_llm.py"
        model_name="THUDM/chatglm-6b"
        input_model="/tf_dataset2/models/pytorch/chatglm-6b"
        extension=" --model_name chatglm --glm_tokenizer $input_model"
        precision_list=("q4_j_b128" "q4_j_b32" "q4_0")
    fi
    echo "=======  Convert Start  ======="
    ## prepare fp32 bin
    python ${convert_script} --outtype f32 --outfile ${working_dir}/${model}-fp32.bin ${input_model}
    echo "=======  Convert End  ======="  
    for cores_per_instance in ${cores_list[@]}
    do
        for batch_size in ${batch_size_list[@]}
        do
            for input in ${input_list[@]}
            do
                for precision in ${precision_list[@]}
                do
                   # [[ "${input}" == "32" ]] && output=32 ||
			        output=32
                    if [[ "${input}" == "32" ]]; then
                        if [[ "${model}" == "chatglm2" || "${model}" == "chatglm-6b" ]]; then
                            prompt='"question": "专注女性情感，成长，智慧\n### 上面这句话可以划分到下面哪一类？\n### 类别：其它，低俗，引流，赌博，辱骂，壮阳\n### 答案：\n"'
                        elif [[ "${model}" == "llama"* ]]; then
                            prompt="Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
                        else
                            prompt="Once upon a time, there existed a little girl, who liked to have adventures. She wanted to go to places and meet new people, and have fun."
                        fi
                    elif [[ "${input}" == "100" ]]; then
                        prompt="Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun.And one day she went with her mother in search of adventure to an exotic land unknown to them, but they never returned. The next day, her relatives started searching for her, and that is how the little girl ended up in a strange town called San Diego, with no family or friends, and a lot of "
                    elif [[ "${input}" == "2012" ]]; then
                        if [[ "${model}" == "chatglm2" || "${model}" == "chatglm-6b" ]]; then
                            prompt="你好"
                        elif [[ "${model}" == "llama"* ]]; then
			                prompt="It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I have not seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate, it starts conquering it for its team. It takes around 10 seconds for a plate to be conquered; less if more pasta from the same team are around. If pasta from other team are around, though, they get locked down in their attempt, unable to conquer the plate, until one of them die (think Battlefield's standard 'Conquest' mode). You get points every second for every plate you own. Over time, the concept also evolved to use an Italian bistro as its main scenario. Carlos, Carlos' Bistro's founder and owner Setup No major changes were made from my work setup. I used FDT and Starling creating an Adobe AIR (ActionScript) project, all tools or frameworks I already had some knowledge with. One big change for me was that I livestreamed my work through a twitch.tv account. This was a new thing for me. As recommended by Roushey, I used a program called XSplit and I got to say, it is pretty amazing. It made the livestream pretty effortless and the features are awesome, even for the free version. It was great to have some of my friends watch me, and then interact with them and random people through chat. It was also good knowing that I was also recording a local version of the files, so I could make a timelapse video later. Knowing the video was being recorded also made me a lot more self-conscious about my computer use, as if someone was watching over my shoulder. It made me realize that sometimes I spend too much time in seemingly inane tasks (I ended up wasting the longest time just to get some text alignment the way I wanted - it'll probably drive someone crazy if they watch it) and that I do way too many typos where writing code. I pretty much spend half of the time writing a line and the other half fixing the crazy characters in it. My own stream was probably boring to watch since I was coding for the most time. But livestreaming is one of the cool things to do as a spectator too. It was great seeing other people working - I had a few tabs opened on my second monitor all the time. It's actually a bit sad, because if I could, I could have spent the whole weekend just watching other people working! But I had to do my own work, so I'd only do it once in a while, when resting for a bit. Design Although I wanted some simple, low-fi, high-contrast kind of design, I ended up going with somewhat realistic (vector) art. I think it worked very well, fitting the mood of the game, but I also went overboard. For example: to know the state of a plate (who owns it, who's conquering it and how much time they have left before conquering it, which pasta units are in the queue, etc), you have to look at the plate's bill. The problem I realized when doing some tests is that people never look at the bill! They think it's some kind of prop, so they never actually read its details. Plus, if you're zoomed out too much, you can't actually read it, so it's hard to know what's going on with the game until you zoom in to the area of a specific plate. One other solution that didn't turn out to be as perfect as I thought was how to indicate who a plate base belongs to. In the game, that's indicated by the plate's decoration - its color denotes the team owner. But it's something that fits so well into the design that people never realized it, until they were told about it. In the end, the idea of going with a full physical metaphor is one that should be done with care. Things that are very important risk becoming background noise, unless the player knows its importance. Originally, I wanted to avoid any kind of heads-up display in my game. In the end, I ended up adding it at the bottom to indicate your credits and bases owned, as well as the hideous out-of-place-and-still-not-obvious 'Call Waiter' button. But in hindsight, I should have gone with a simple HUD from the start, especially one that indicated each team's colors and general state of the game without the need for zooming in and out. Development Development went fast. But not fast enough. Even though I worked around 32+ hours for this Ludum Dare, the biggest problem that I had to face in the end was overscoping."
                        elif [[ "${model}" == "gptj-6b" ]]; then
                            prompt="It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I haven't seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level, but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate, it starts conquering it for its team. It takes around 10 seconds for a plate to be conquered; less if more pasta from the same team are around. If pasta from other team are around, though, they get locked down in their attempt, unable to conquer the plate, until one of them die (think Battlefield's standard 'Conquest' mode). You get points every second for every plate you own. Over time, the concept also evolved to use an Italian bistro as its main scenario. Carlos, Carlos' Bistro's founder and owner Setup No major changes were made from my work setup. I used FDT and Starling creating an Adobe AIR (ActionScript) project, all tools or frameworks I already had some knowledge with. One big change for me was that I livestreamed my work through a twitch.tv account. This was a new thing for me. As recommended by Roushey, I used a program called XSplit and I got to say, it is pretty amazing. It made the livestream pretty effortless and the features are awesome, even for the free version. It was great to have some of my friends watch me, and then interact with them and random people through chat. It was also good knowing that I was also recording a local version of the files, so I could make a timelapse video later. Knowing the video was being recorded also made me a lot more self-conscious about my computer use, as if someone was watching over my shoulder. It made me realize that sometimes I spend too much time in seemingly inane tasks (I ended up wasting the longest time just to get some text alignment the way I wanted - it'll probably drive someone crazy if they watch it) and that I do way too many typos where writing code. I pretty much spend half of the time writing a line and the other half fixing the crazy characters in it. My own stream was probably boring to watch since I was coding for the most time. But livestreaming is one of the cool things to do as a spectator too. It was great seeing other people working - I had a few tabs opened on my second monitor all the time. It's actually a bit sad, because if I could, I could have spent the whole weekend just watching other people working! But I had to do my own work, so I'd only do it once in a while, when resting for a bit. Design Although I wanted some simple, low-fi, high-contrast kind of design, I ended up going with somewhat realistic (vector) art. I think it worked very well, fitting the mood of the game, but I also went overboard. For example: to know the state of a plate (who owns it, who's conquering it and how much time they have left before conquering it, which pasta units are in the queue, etc), you have to look at the plate's bill. The problem I realized when doing some tests is that people never look at the bill! They think it's some kind of prop, so they never actually read its details. Plus, if you're zoomed out too much, you can't actually read it, so it's hard to know what's going on with the game until you zoom in to the area of a specific plate. One other solution that didn't turn out to be as perfect as I thought was how to indicate who a plate base belongs to. In the game, that's indicated by the plate's decoration - its color denotes the team owner. But it's something that fits so well into the design that people never realized it, until they were told about it. In the end, the idea of going with a full physical metaphor is one that should be done with care. Things that are very important risk becoming background noise, unless the player knows its importance. Originally, I wanted to avoid any kind of heads-up display in my game. In the end, I ended up adding it at the bottom to indicate your credits and bases owned, as well as the hideous out-of-place-and-still-not-obvious 'Call Waiter' button. But in hindsight, I should have gone with a simple HUD from the start, especially one that indicated each team's colors and general state of the game without the need for zooming in and out. Development Development went fast. But not fast enough. Even though I worked around 32+ hours for this Ludum Dare, the biggest problem I had to face in the end was overscoping. I had too much planned, and could not get it all done. Content-wise, I had several kinds of pasta planned - Wikipedia is just amazing in that regard, split into several different groups, from small Pastina to huge Pasta al forno. But because of time constraints, I ended up scratching most of them, and ended up with 5 different types of small pasta - barely something to start when talking about the evolution of Pasta. Pastas used in the game. Unfortunately, the macs where never used Which is one of the saddest things about the project, really. It had the framework and the features to allow an endless number of elements in there, but I just did not have time to draw the rest of the assets needed (something I loved to do)."
                        else
                            prompt="It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on web. Playing on web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I haven't seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level, but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate, it starts conquering it for its team. It takes around 10 seconds for a plate to be conquered; less if more pasta from the same team are around. If pasta from other team are around, though, they get locked down in their attempt, unable to conquer the plate, until one of them die (think Battlefield's standard 'Conquest' mode). You get points every second for every plate you own. Over time, the concept also evolved to use an Italian bistro as its main scenario. Carlos, Carlos' Bistro's founder and owner Setup No major changes were made from my work setup. I used FDT and Starling creating an Adobe AIR (ActionScript) project, all tools or frameworks I already had some knowledge with. One big change for me was that I livestreamed my work through a twitch.tv account. This was a new thing for me. As recommended by Roushey, I used a program called XSplit and I got to say, it is pretty amazing. It made the livestream pretty effortless and the features are awesome, even for the free version. It was great to have some of my friends watch me, and then interact with them and random people through chat. It was also good knowing that I was also recording a local version of the files, so I could make a timelapse video later. Knowing the video was being recorded also made me a lot more self-conscious about my computer use, as if someone was watching over my shoulder. It made me realize that sometimes I spend too much time in seemingly inane tasks (I ended up wasting the longest time just to get some text alignment the way I wanted - it'll probably drive someone crazy if they watch it) and that I do way too many typos where writing code. I pretty much spend half of the time writing a line and the other half fixing the crazy characters in it. My own stream was probably boring to watch since I was coding for the most time. But livestreaming is one of the cool things to do as a spectator too. It was great seeing other people working - I had a few tabs opened on my second monitor all the time. It's actually a bit sad, because if I could, I could have spent the whole weekend just watching other people working! But I had to do my own work, so I'd only do it once in a while, when resting for a bit. Design Although I wanted some simple, low-fi, high-contrast kind of design, I ended up going with somewhat realistic (vector) art. I think it worked very well, fitting the mood of the game, but I also went overboard. For example: to know the state of a plate (who owns it, who's conquering it and how much time they have left before conquering it, which pasta units are in the queue, etc), you have to look at the plate's bill. The problem I realized when doing some tests is that people never look at the bill! They think it's some kind of prop, so they never actually read its details. Plus, if you're zoomed out too much, you can't actually read it, so it's hard to know what's going on with the game until you zoom in to the area of a specific plate. One other solution that didn't turn out to be as perfect as I thought was how to indicate who a plate base belongs to. In the game, that's indicated by the plate's decoration - its color denotes the team owner. But it's something that fits so well into the design that people never realized it, until they were told about it. In the end, the idea of going with a full physical metaphor is one that should be done with care. Things that are very important risk becoming background noise, unless the player knows its importance. Originally, I wanted to avoid any kind of heads-up display in my game. In the end, I ended up adding it at the bottom to indicate your credits and bases owned, as well as the hideous out-of-place-and-still-not-obvious 'Call Waiter' button. But in hindsight, I should have gone with a simple HUD from the start, especially one that indicated each team's colors and general state of the game without the need for zooming in and out. Development Development went fast. But not fast enough. Even though I worked around 32+ hours for this Ludum Dare, the biggest problem I had to face in the end was overscoping. I had too much planned, and could not get it all done. Content-wise, I had several kinds of pasta planned - Wikipedia is just amazing in that regard, split into several different groups, from small Pastina to huge Pasta al forno. But because of time constraints, I ended up scratching most of them, and ended up with 5 different types of small pasta - barely something to start when talking about the evolution of Pasta. Pastas used in the game. Unfortunately, the macs where never used Which is one of the saddest things about the project, really. It had the framework and the features to allow an endless number of"
                        fi
                    elif [[ "${input}" == "1024" ]]; then
                        if [[ "${model}" == "chatglm2" || "${model}" == "chatglm-6b" ]]; then
                            prompt='"question": "我再拍一单给我妹妹\n### 上面这句话可以划分到下面哪一类？\n### 类别：其它，低俗，引流，赌博，辱骂，壮阳\n### 答案：\n"'
                        elif [[ "${model}" == "llama"* ]]; then
                            prompt="It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I have not seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate,",
                        elif [[ "${model}" == "gptj-6b" ]]; then
                            prompt="It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I haven't seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level, but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate, it starts conquering it for its team. It takes around 10 seconds for a plate to be conquered; less if more pasta from the same team are around. If pasta from other team are around, though, they get locked down in their attempt, unable to conquer the plate, until one of them die (think Battlefield's standard 'Conquest' mode). You get points every second for every plate you own. Over time, the concept"
                        else
                            prompt="It is done, and submitted. You can play 'Survival of the Tastiest' on the Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in the space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face it. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I haven't seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level, but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in the vicinity of a plate, it starts conquering it for its team. It takes around 10 seconds for a plate to be conquered; less if more pasta from the same team are around."
                        fi
                    fi
                    ctx=$(($output + $input + 10))
                    logs_file="${model}-${precision}-${cores_per_instance}-${batch_size}-${input}-${output}.log"
                    ## prepare model.bin
                    echo "=======  Quantization Start  ======="
                    if [[ ${precision} == "q4_j_vnni_b128" ]]; then
                        ${quant_script} --model_file ${working_dir}/${model}-fp32.bin --out_file ${working_dir}/${model}-${precision}.bin --bits 4 --block_size 128 --scale_dtype fp32 --compute_type int8 --alg sym
                    elif [[ ${precision} == "q4_j_vnni_bf16_b32" ]]; then
                        ${quant_script} --model_file ${working_dir}/${model}-fp32.bin --out_file ${working_dir}/${model}-${precision}.bin --bits 4 --block_size 32 --scale_dtype bf16 --compute_type int8 --alg sym
                    elif [[ ${precision} == "q4_j_vnni_b32" ]]; then
                        ${quant_script} --model_file ${working_dir}/${model}-fp32.bin --out_file ${working_dir}/${model}-${precision}.bin --bits 4 --block_size 32 --scale_dtype fp32 --compute_type int8 --alg sym
                    elif [[ ${precision} == "q4_j_b32" ]]; then
                        ${quant_script} --model_file ${working_dir}/${model}-fp32.bin --out_file ${working_dir}/${model}-${precision}.bin --bits 4 --block_size 32 --scale_dtype fp32 --compute_type int8 --alg sym
                    elif [[ ${precision} == "q4_j_b128" ]]; then
                        ${quant_script} --model_file ${working_dir}/${model}-fp32.bin --out_file ${working_dir}/${model}-${precision}.bin --bits 4 --block_size 128 --scale_dtype fp32 --compute_type int8 --alg sym
                    elif [[ ${precision} == "q4_j_channel" ]]; then
                        ${quant_script} --model_file ${working_dir}/${model}-fp32.bin --out_file ${working_dir}/${model}-${precision}.bin --bits 4 --block_size -1 --scale_dtype fp32 --compute_type int8 --alg sym
                    elif [[ ${precision} == "q4_j_b128_asym" ]]; then
                        ${quant_script} --model_file ${working_dir}/${model}-fp32.bin --out_file ${working_dir}/${model}-${precision}.bin --bits 4 --block_size 128 --scale_dtype fp32 --compute_type int8 --alg asym
                    elif [[ ${precision} == "q4_0" ]]; then    
                        ${quant_script} --model_file ${working_dir}/${model}-fp32.bin --out_file ${working_dir}/${model}-${precision}.bin --bits 4 --block_size 32 --compute_type ggml --alg sym
                    elif [[ ${precision} == "q4_1" ]]; then    
                        ${quant_script} --model_file ${working_dir}/${model}-fp32.bin --out_file ${working_dir}/${model}-${precision}.bin --bits 4 --block_size 32 --compute_type ggml --alg asym
                    elif [[ ${precision} == "q8_0" ]]; then
                        ${quant_script} --model_file ${working_dir}/${model}-fp32.bin --out_file ${working_dir}/${model}-${precision}.bin --bits 8 --block_size 32 --compute_type ggml --alg sym
                    elif [[ ${precision} == "q8_j_b128" ]]; then
                        ${quant_script} --model_file ${working_dir}/${model}-fp32.bin --out_file ${working_dir}/${model}-${precision}.bin --bits 8 --block_size 128  --scale_dtype fp32 --compute_type int8 --alg sym 
                    elif [[ ${precision} == "q8_j_channel" ]]; then
                        ${quant_script} --model_file ${working_dir}/${model}-fp32.bin --out_file ${working_dir}/${model}-${precision}.bin --bits 8 --block_size -1  --scale_dtype fp32 --compute_type int8 --alg sym
                    fi
                    echo "=======  Quantization End  ======="
                    ## run inference
                    export LANG=en_US.UTF-8
                    export LC_ALL=en_US.UTF-8
                    #export cores_per_instance=${cores_per_instance}
                    #infer_cmd="${infer_cmd} --seed 1234 --keep -1 -t 32 --repeat_penalty 1.0 --color -c ${ctx} -n ${output} -m ${model}-${precision}.bin -p \"$promt\" "
                    #OMP_NUM_THREADS=$[$cores_per_instance * 1] numactl -m 0 -C 0-$[$cores_per_instance * 1 - 1] \
                    #python ${validation_script} --command ${infer_cmd} --input_tokens ${input} --output_token ${output} --model ${model}-${precision}.bin 2>&1 |tee ./${logs_file} || true
                    #python ${validation_script} --command ${infer_cmd} --input_tokens ${input} --output_token ${output} --model ${model}-${precision}.bin --log_file ${logs_file}
                    echo "=======  Inference Start  ======="
                    if [[ "${model}" == "chatglm2" || "${model}" == "chatglm-6b" ]]; then
                        OMP_NUM_THREADS=$[$cores_per_instance * 1] numactl -m 0 -C 0-$[$cores_per_instance * 1 - 1] \
                        $infer_cmd  --seed 1234 -b 2047 -t $cores_per_instance -m ${model}-${precision}.bin $extension -p "$prompt" 2>&1 |tee ./${logs_file} || true&minitor
                    else
                        OMP_NUM_THREADS=$[$cores_per_instance * 1] numactl -m 0 -C 0-$[$cores_per_instance * 1 - 1] \
                        $infer_cmd  --seed 1234 -b 2047 -t $cores_per_instance -c ${ctx} -n ${output} -m ${model}-${precision}.bin $extension -p "$prompt" 2>&1 |tee ./${logs_file} || true&minitor
                    fi
                    rm  ${model}-${precision}.bin
                    echo "=======  Inference End  ======="
		            python  ./calculate_percertiles.py ./${logs_file} ${model} ${precision} ${cores_per_instance} ${batch_size} ${input} ${output}
                done
		        #numactl -C 0 python calculate_percertiles.py ${logs_file} ${model} ${precision} ${cores_per_instance} ${batch_size} ${input} ${output}
            done
        done
    done
}


function collect_perf_logs_llm {
    # latency
    log_dir="./$1"
    eval_time=($(grep -i 'eval time' ${log_dir} | grep -v "prompt" | sed -e 's/.*eval time = .* runs.*(//;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
    total_time=($(grep -i 'total time' ${log_dir} | sed -e 's/.*total time = //;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
    first_token_time=($(grep -i 'eval time' ${log_dir} | grep "prompt" | sed -e 's/.*prompt eval time = .* tokens.*(//;s/[^0-9.]//g;s/\.$//' |awk '
        BEGIN {
            num = 0;
            sum = 0;
        }{
            num ++;
            sum += $1;
        }END {
            if(num > 0) {
                printf("%.6f", sum / num);
            }else {
                printf("0");
            }
        }
    '))
    input_tokens=$input
    max_new_tokens=$output
    # memory usage
    used_memory=$(grep 'memory used total:' ${log_dir} |tail -n 1 |head -n 1 |awk '{print $(NF-1)}')
    # summary
    framework="engine"
    mode_name="latency"
    precision=$2
    link="${log_prefix}/$1"
    printf "${framework},${mode_name},${model_name},${precision},${batch_size}," |tee -a ${WORKSPACE}/cpp_graph_summary.log
    printf "${input_tokens},${max_new_tokens},${cores_per_instance},${latency[1]}," |tee -a ${WORKSPACE}/cpp_graph_summary.log
    printf "${first_token_time},${eval_time},${total_time},${used_memory},${link}\n" |tee -a ${WORKSPACE}/cpp_graph_summary.log
    set +x
    echo -e "\n\n-------- Summary --------"
    sed -n '1p;$p' ./cpp_graph_summary.log |column -t -s ','
}

function minitor() {
        sleep 2
        echo "======  Monitor Start ======="
        while true
        do
            if [ $(ps -ef |grep "$infer_cmd" |wc -l) -lt 2 ];then
			    #python calculate_percertiles.py ${logs_file} ${model} ${precision} ${cores_per_instance} ${batch_size} ${input} ${output}
			    sleep 3
                break
            fi
            echo "$(date +%s), $(numastat -p $(ps -ef |grep "$infer_cmd" |grep -v grep |awk '{printf("%s  ", $2)}'))">> ./memory.txt 2>&1
        done
        echo "======  Monitor End ======="
}
function get_data() {
	python calculate_percertiles.py ${logs_file} ${model} ${precision} ${cores_per_instance} ${batch_size} ${input} ${output} 
}
main $@ 2>&1 |tee $./launch.log