import projectsData from '@/data/projectsData'

import ProjectCard from '@/components/ProjectCard'
import AnimatedDiv from '@/components/framer-motion/AnimatedDiv'
import { FadeContainer } from '../lib/FramerMotionVariants'
import Link from '@/components/Link'
import { useState, useEffect } from 'react'
import { SERVER_BASE_URL } from '@/data/siteMetadata'

const RecentProjects = ({ MAX_PROJECTS }) => {

  const [projects, setProjects] = useState([]);

  useEffect(() => {
      fetch(`${SERVER_BASE_URL || 'http://localhost:8000'}/api/projects/`)
          .then(response => response.json())
          .then(data => {
              const projectsList = data.slice(0, MAX_PROJECTS);
              //console.log(projectsList);
              setProjects(projectsList);
          })
          .catch(error => console.error('Error fetching projects:', error));
  }, []) // The empty array means this effect runs once on component mount.

  return (
    <>
      <div className="divide-y divide-gray-700">
        <div className="my-4">
          <span className="font-poppins title-font text-3xl font-bold">最近的项目</span>
        </div>
        <div className="py-5">
          <AnimatedDiv
            variants={FadeContainer}
            className="mx-auto grid grid-cols-1 gap-4 md:ml-[20%] xl:ml-[24%]"
          >
            {projects.map((d) => (
              <ProjectCard
                key={d.title}
                title={d.title}
                description={d.description}
                imgSrc={d.imgSrc}
                href={d.href}
                tools={d.tools}
                deployed={d.deployed}
              />
            ))}
          </AnimatedDiv>
        </div>
        <div className="mt-5 flex justify-end text-base font-medium leading-6">
          <Link href="/projects" className="mt-5 hover:text-primary-400" aria-label="all posts">
            所有项目 &rarr;
          </Link>
        </div>
      </div>
    </>
  )
}

export default RecentProjects
